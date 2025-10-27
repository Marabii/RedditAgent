#!/usr/bin/env node
"use strict";

const { LMStudioClient } = require("@lmstudio/sdk");
const { z } = require("zod");
const path = require("path");

const DEFAULT_PROMPT =
  "You are analysing screenshots of reddit posts. Carefully inspect both text and visuals of the provided image. Your job is to determine whether the post is an advertisment (ad) or just a regular post, identify the primary language: return true if the text in this image is written in arabic otherwise return false, and provide a concise but complete description of its visible content. Explicitly decide whether the post text is predominantly Arabic. Return ONLY JSON with keys: isAd (boolean), isArabic (boolean), language (string), description (string), confidence (number between 0 and 1).";

const responseSchema = z.object({
  isAd: z.boolean(),
  description: z.string().min(1, "Description must not be empty"),
  confidence: z
    .number({ invalid_type_error: "Confidence must be numeric" })
    .min(0)
    .max(1)
    .optional(),
  isArabic: z.boolean().default(false),
  language: z.string().min(1, "Language must be provided").default("unknown"),
});

const responseJsonSchema = {
  type: "object",
  properties: {
    isAd: { type: "boolean" },
    description: { type: "string", minLength: 1 },
    confidence: {
      type: "number",
      minimum: 0,
      maximum: 1,
    },
    isArabic: { type: "boolean" },
    language: { type: "string" },
  },
  required: ["isAd", "description", "isArabic", "language"],
  additionalProperties: false,
};

function printUsage(exitCode = 1) {
  console.error(
    "Usage: node extract_post_info.cjs <image-path> [--prompt 'optional question']"
  );
  process.exit(exitCode);
}

function parseArgs(argv) {
  if (!argv.length) {
    printUsage();
  }

  const args = { imagePath: null, prompt: DEFAULT_PROMPT };
  const rest = [...argv];

  args.imagePath = rest.shift();
  while (rest.length) {
    const flag = rest.shift();
    if (flag === "--prompt") {
      if (!rest.length) {
        console.error("--prompt requires a value");
        printUsage();
      }
      args.prompt = rest.shift();
    } else {
      console.error(`Unknown argument: ${flag}`);
      printUsage();
    }
  }

  if (!args.imagePath) {
    printUsage();
  }

  return args;
}

async function runAnalysis(imagePath, promptText) {
  const client = new LMStudioClient();
  const resolvedPath = path.resolve(imagePath);
  const preparedImage = await client.files.prepareImage(resolvedPath);

  // Prefer an explicitly requested DeepSeek-compatible multimodal model name if available.
  const modelName = "qwen/qwen2.5-vl-7b";
  const model = await client.llm.model(modelName);

  const response = await model.respond(
    [
      {
        role: "user",
        content: promptText,
        images: [preparedImage],
      },
    ],
    {
      maxTokens: 512,
      structured: { type: "json", jsonSchema: responseJsonSchema },
    }
  );

  const parsed = responseSchema.parse(JSON.parse(response.content));
  return {
    ...parsed,
    model: modelName,
    prompt: promptText,
    imagePath: resolvedPath,
  };
}

(async () => {
  try {
    const { imagePath, prompt } = parseArgs(process.argv.slice(2));
    const result = await runAnalysis(imagePath, prompt);
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error(
      JSON.stringify(
        {
          error: error?.message || String(error),
          stack: error?.stack,
        },
        null,
        2
      )
    );
    process.exit(1);
  }
})();
