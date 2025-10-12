// extract_post_info.cjs
const { LMStudioClient } = require("@lmstudio/sdk");
const path = require("path");

const schema = {
  type: "object",
  properties: {
    isAd: { type: "boolean" },
    description: { type: "string" }
  },
  required: ["isAd", "description"],
  additionalProperties: false
};

(async () => {
  const client = new LMStudioClient();
  const model = await client.llm.model();

  const img = await client.files.prepareImage(
    path.resolve("/home/hamza/Desktop/Coding/Projects/AIProject/RedditAgent/tools/temporary/posts/post_20251010_200729.png")
  );

  const res = await model.respond(
    [
      {
        role: "user",
        content:
          'You are analysing screenshots of reddit posts. Carefully inspect both text and visuals of the provided image. Your job is to determine whether the post is an advertisment (ad) or just a regular post and to provide a concise but complete description of its visible content. Return ONLY JSON that follows this schema: {"isAd": boolean, "description": string}.',
        images: [img]
      }
    ],
    { structured: { type: "json", jsonSchema: schema }, maxTokens: 300 }
  );

  const data = JSON.parse(res.content);
  console.log(data);
})();
