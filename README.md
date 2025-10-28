# Reddit Agent Automation Suite

An automation pipeline that navigates Reddit, discovers valuable posts, reasons about them, and drafts context-aware comments. The system combines a fine-tuned GroundingDINO detector, a LangGraph-driven orchestration layer (powered by DeepSeek R1), and a toolbox of Python utilities that capture, classify, and interact with the live Reddit UI.

This project was built to demonstrate full-stack applied ML engineering—data collection, model finetuning, tool wiring, and autonomous agent design.

---

## ✨ Highlights

- **Custom GroundingDINO model** trained on thousands of Reddit screenshots automatically annotated in COCO format (RedditUiScraper)[https://github.com/Marabii/RedditUiScraper], yielding precise detection of Reddit UI elements (post cards, titles, upvote/downvote, comment buttons, composers, etc.).
- **End-to-end agent loop** that captures posts, skips ads and duplicates, reasons about helpful actions, pastes comments, and resumes the feed—while keeping the operator in the loop.
- **Resilient tooling** in `tools/` for screenshot capture, OCR, region validation, comment submission, Perplexity summaries, and more... Each callable independently or orchestrated together.
- **Deep logging & manual safeguards** including kill switches (`e`), manual resume (`r`), and structured history so you always know what the agent is doing.

---

## 🏗 System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                   LangGraph Orchestrator                   │
│  (DeepSeek R1 reasoning ➜ plan ➜ act ➜ observe ➜ loop)    │
└───────────────▲───────────────────────────▲───────────────┘
                │                           │
                │                           │
        ┌───────┴───────┐           ┌───────┴────────┐
        │  Tools Layer  │           │  Tool Outputs  │
        │  (Python)     │           │  (JSON, Images) │
        └───────▲───────┘           └────────▲────────┘
                │                           │
   ┌────────────┴──────────┐  ┌─────────────┴───────────┐
   │ GroundingDINO (FT)    │  │ LM Studio / DeepSeek R1 │
   │ Object Detection      │  │ Perplexity (optional)    │
   └───────────────────────┘  └──────────────────────────┘
```

<img width="349" style="margin: 0 auto" height="1222" alt="reddit_agent_graph" src="https://github.com/user-attachments/assets/44d58c54-1e36-4d77-a958-521bec70bfd1" />

### Key components

| Component | Purpose |
|-----------|---------|
| `tools/getPost.py` | Captures the next visible post, validates bounding boxes, skips duplicates, and writes metadata (`lastPostInfo.json`, `recentTitles.json`) all using the fine-tuned GroundingDino model. |
| `tools/extract_post_info.cjs` | Runs Qwen2-VL model via LM Studio to determine whether a post is an ad and to summarize its content. |
| `tools/reddit_commenter.py` | Locates the comment composer using GroundingDINO, pastes a generated response, and leaves it ready for manual submission. |
| `tools/prompt_powerful_ai.py` | Captures detailed post crops (`postCardInDetail`) and (optionally) sends them to Perplexity for a rich explanation. |
| `tools/reddit_langgraph_agent.py` | Orchestrates the full automation loop with LangGraph, DeepSeek R1 reasoning, and a manual resume/killswitch. |
| `GroundingDino/` | Hosts the fine-tuned GroundingDINO weights, configs, and helper scripts. |

---

## 🧠 Fine-tuning the Detector

1. **Data collection**: used Puppeteer to scrape Reddit’s feed, capturing thousands of screenshots across multiple layouts.
2. **Auto annotation**: generated COCO-format labels for UI elements and post structure, enabling supervised detection without manual labeling.
3. **GroundingDINO fine-tuning**: trained on the dataset to specialize the model for Reddit’s interface. The project now ships with that checkpoint, delivering robust detections of:
   - `postCard`, `postTitle`
   - `upvote`, `downvote`, `comment`
   - `postCardInDetail`, `commentComposer`, `submitComment`, `cancelComment`

This detector is the backbone for every automation step: post selection, composer validation, and screenshot cropping.

---

## 🔁 Agent Workflow (tools/reddit_langgraph_agent.py)

1. **Plan** – DeepSeek R1 creates/refreshes a short plan for the user’s objective.
2. **Fetch Post** – `getPost.capture_next_post()` captures the next valid post, skipping ads/duplicates.
3. **Extract & Classify** – `extract_post_info.cjs` (Qwen2-VL via LM Studio) returns `isAd`, `isArabic`, `description`, etc.
4. **Decide** – DeepSeek R1 chooses the best action: upvote, downvote, comment, or skip.
5. **Act** – Invokes the relevant tool (e.g., `action_taker.upvote_post`). Commenting jumps into the composer page.
6. **Describe** – Depending on config, either Perplexity or DeepSeek summarizes the detailed post view.
7. **Compose** – DeepSeek crafts a context-aware, empathetic reply aligned with the initial objective.
8. **Submit** – `write_reddit_comment` pastes the comment. The operator presses `r` when done (submit or cancel).
9. **Navigate Back** – `Alt + Left` returns to the home feed; the loop repeats until stopped or the loop count is reached.

### Manual controls

| Key | Action |
|-----|--------|
| `e` | Kill switch: immediately stops the agent and future loops. |
| `r` | Resume after reviewing/publishing the drafted comment. |

---

## ⚙️ Prerequisites

- **Python** 3.11+ (project tested on 3.12).
- **Node.js** 18+ (for the LM Studio SDK script in `extract_post_info.cjs`).
- `pip install -r requirements.txt` *(create one or install dependencies manually: `torch`, `mss`, `Pillow`, `pynput`, `requests`, `langgraph`, etc.)*
- LM Studio running locally with:
  - Qwen/Qwen2-VL-7B-Instruct
  - DeepSeek R1 reasoning model exposed via OpenAI-compatible endpoint (`LMSTUDIO_BASE_URL`).
- Perplexity API key (optional). Set `REDDIT_AGENT_DESCRIBE_MODE=local` to stay fully offline.
- Fine-tuned GroundingDino
---

## 🚀 Quick Start

1. **Clone & install**
   ```bash
   git clone https://github.com/<your-handle>/RedditAgent.git
   cd RedditAgent
   pip install -r requirements.txt
   npm install --prefix tools
   ```

2. **Environment variables** (edit `.env` or export):
   ```bash
   export LMSTUDIO_BASE_URL="http://127.0.0.1:1234/v1"
   export LMSTUDIO_MODEL="deepseek/deepseek-r1-0528-qwen3-8b"
   export LMSTUDIO_API_KEY="Dattebayo"            # or your token
   export PERPLEXITY_API_KEY="sk-..."             # optional
   export REDDIT_AGENT_DESCRIBE_MODE="perplexity" # or "local"
   ```

3. **Capture a single post**
   ```bash
   python tools/getPost.py
   ```
   Resulting artifacts land in `tools/temporary/posts`. Metadata lives in `tools/temporary/lastPostInfo.json` and `recentTitles.json`.

4. **Run the LangGraph agent**
   ```bash
   python tools/reddit_langgraph_agent.py "boost my karma by making helpful comments" 3
   ```
   - When a comment is ready, review it and hit **`r`** to resume.
   - Press **`e`** at any time to stop.

5. **(Optional) Describe a post crop**
   ```bash
   python tools/prompt_powerful_ai.py --monitor 0 --describe --prompt "Explain this post"
   ```

---

## 🔍 Logging & Debugging

- Each tool prints explicit status messages. Set `SAVE_FAILED_ATTEMPTS=True` in `getPost.py` if you want annotated debugging images of missed detections.
- The agent logs every node transition, decision, and manual latch state, producing a clear timeline of actions.
- `recentTitles.json` prevents capturing the same content repeatedly; delete it if you want a cold start.

---

## 📁 Project Structure

```
.
├── GroundingDino/                # Fine-tuned detector + utilities
├── tools/
│   ├── action_taker.py           # Upvote/downvote/comment button clickers
│   ├── extract_post_info.cjs     # LM Studio + Qwen2-VL ad classifier
│   ├── getPost.py                # Post capture logic (titles, deduping)
│   ├── prompt_powerful_ai.py     # Detailed crop + Perplexity bridge
│   ├── reddit_commenter.py       # Comment composer automation
│   ├── reddit_langgraph_agent.py # LangGraph orchestrator
│   └── temporary/                # Runtime artifacts (screens, metadata)
├── README.md
└── ...
```
