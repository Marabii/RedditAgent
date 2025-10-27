#!/usr/bin/env python3
"""LangGraph-based Reddit automation agent orchestrated by DeepSeek R1."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypedDict
from uuid import uuid4

import requests
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pynput import keyboard as pynput_keyboard

from action_taker import comment_on_post, downvote_post, upvote_post
from getPost import capture_next_post
from prompt_powerful_ai import (
    capture_postcard_detail,
    describe_post_image,
)
from reddit_commenter import write_reddit_comment


logger = logging.getLogger("reddit_agent")
keyboard_controller = pynput_keyboard.Controller()


THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_think_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = THINK_TAG_PATTERN.sub("", text)
    return cleaned.strip()


def to_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]
    return value


def navigate_back_to_feed(reason: Optional[str] = None, delay: float = 0.4) -> None:
    message = "Navigating back to feed"
    if reason:
        message += f" ({reason})"
    logger.info(message)
    with keyboard_controller.pressed(pynput_keyboard.Key.alt):
        keyboard_controller.press(pynput_keyboard.Key.left)
        keyboard_controller.release(pynput_keyboard.Key.left)
    time.sleep(max(0.0, delay))


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        root.setLevel(level)


TOOLS_ROOT = Path(__file__).resolve().parent
TEMP_DIR = TOOLS_ROOT / "temporary"
POST_DIR = TEMP_DIR / "posts"
EXTRACT_SCRIPT = TOOLS_ROOT / "extract_post_info.cjs"


class AgentState(TypedDict, total=False):
    objective: str
    plan: List[str]
    last_plan_refresh: float
    current_step_index: int
    loop_count: int
    max_loops: int
    post_payload: Dict[str, Any]
    post_analysis: Dict[str, Any]
    skip_reason: Optional[str]
    decision: Optional[Literal["upvote", "downvote", "comment", "skip"]]
    description: Optional[str]
    composed_comment: Optional[str]
    last_detail_image: Optional[str]
    history: List[str]
    terminated: bool
    last_error: Optional[str]


@dataclass
class KillSwitch:
    """Keyboard-based kill switch triggered by pressing the 'e' key."""

    def __post_init__(self) -> None:
        self._event = threading.Event()
        self._listener = pynput_keyboard.Listener(on_press=self._on_press)

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()
        try:
            self._listener.join()
        except RuntimeError:
            pass

    def triggered(self) -> bool:
        return self._event.is_set()

    def _on_press(self, key: pynput_keyboard.Key) -> None:
        if getattr(key, "char", "") == "e":
            if not self._event.is_set():
                logger.warning("Kill switch triggered via 'e' key")
            self._event.set()


@dataclass
class ContinueWaiter:
    """Pause execution until the user presses the 'c' key to continue."""

    hotkey: str = "c"

    def __post_init__(self) -> None:
        self._event = threading.Event()
        self._listener = pynput_keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def wait(self, kill_switch: Optional[KillSwitch] = None, poll: float = 0.1) -> bool:
        """Block until the continue key is pressed or the kill switch fires.

        Returns True when the continue key is received. Returns False if the
        kill switch is triggered before that happens.
        """

        logger.info("Waiting for user to press '%s' to continue", self.hotkey)
        while True:
            if kill_switch and kill_switch.triggered():
                logger.info("Kill switch triggered while waiting for continue key")
                return False
            if self._event.wait(timeout=poll):
                self._event.clear()
                logger.info("Detected '%s' keypress; resuming", self.hotkey)
                return True

    def stop(self) -> None:
        self._listener.stop()
        try:
            self._listener.join()
        except RuntimeError:
            pass

    def _on_press(self, key: pynput_keyboard.Key) -> None:
        if getattr(key, "char", "") == self.hotkey:
            self._event.set()


class DeepSeekR1Orchestrator:
    """Wrapper around a locally hosted DeepSeek R1 via LM Studio's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = os.getenv("LMSTUDIO_BASE_URL") or "http://127.0.0.1:1234/v1"
        self.api_key = "Dattebayo"
        self.model = "deepseek/deepseek-r1-0528-qwen3-8b"
        self.timeout = timeout

    def _reason(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
        }

        logger.debug(
            "DeepSeek request | model=%s system_len=%d user_len=%d",
            self.model,
            len(system_prompt),
            len(user_prompt),
        )

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - runtime
            raise RuntimeError(f"LM Studio request failed: {exc}") from exc

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices returned from LM Studio: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError(f"Empty completion from LM Studio: {data}")
        cleaned = strip_think_tags(content)
        logger.debug("DeepSeek response (truncated): %s", cleaned[:200])
        return cleaned

    def generate_plan(self, objective: str) -> List[str]:
        instructions = (
            "Draft a concise numbered plan (max 5 steps) for automating Reddit interactions "
            "to satisfy the user objective. Each step should be a short imperative. "
            "Return the plan as a numbered list."
        )
        user_prompt = f"Objective: {objective}\n{instructions}"
        raw = self._reason("You are a planning assistant.", user_prompt)
        return self._extract_bullets(raw)

    def decide_action(self, objective: str, post_summary: Dict[str, Any]) -> str:
        system_prompt = "Select the best action for assisting Reddit users. Reply with one of: upvote, downvote, comment, skip."
        user_payload = json.dumps(post_summary, ensure_ascii=True, indent=2)
        user_prompt = (
            f"Objective: {objective}\nPost Summary JSON:\n{user_payload}\nAction:"
        )
        raw = self._reason(system_prompt, user_prompt).strip().lower()
        for option in ("upvote", "downvote", "comment", "skip"):
            if option in raw:
                return option
        return "skip"

    def compose_comment(self, objective: str, explanation: str) -> str:
        system_prompt = "Write a supportive, empathetic Reddit comment that aligns with the stated objective."
        user_prompt = (
            f"Objective: {objective}\nPost Explanation:\n{explanation}\nComment:"
        )
        result = self._reason(system_prompt, user_prompt).strip()
        return result

    def summarize_post(
        self,
        analysis: Dict[str, Any],
        payload: Dict[str, Any],
        detail_image: Path | str | None = None,
    ) -> str:
        system_prompt = (
            "Summarize the Reddit post in clear, empathetic language. Focus on the main issue, "
            "tone, and any key context that would help craft a helpful reply."
        )
        summary_blob = {
            "analysis": analysis,
            "payload": payload,
            "detail_image": str(detail_image) if detail_image else None,
        }
        user_prompt = (
            "Input JSON about the post:\n"
            f"{json.dumps(summary_blob, ensure_ascii=False, indent=2)}\n"
            "Provide a concise explanation:"
        )
        return self._reason(system_prompt, user_prompt).strip()

    @staticmethod
    def _extract_bullets(text: str) -> List[str]:
        steps: List[str] = []
        for line in text.splitlines():
            line = line.strip(" -*\t")
            if not line:
                continue
            if line[0].isdigit():
                _, _, remainder = line.partition(".")
                steps.append(remainder.strip() or line)
            else:
                steps.append(line)
        return steps or [
            "Review Reddit feed",
            "Select actionable posts",
            "Engage appropriately",
        ]


def with_retries(
    func: Callable[..., Any],
    *args: Any,
    retries: int = 2,
    delay: float = 1.0,
    **kwargs: Any,
) -> Any:
    last_exc: Optional[Exception] = None
    attempts = retries + 1
    name = getattr(func, "__name__", repr(func))
    for attempt in range(1, attempts + 1):
        try:
            logger.info(
                "-------------RETRY LOGIC --------------- Calling %s (attempt %d/%d)",
                name,
                attempt,
                attempts,
            )
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - runtime resilience
            last_exc = exc
            if attempt < attempts:
                logger.warning(
                    "%s failed on attempt %d/%d: %s", name, attempt, attempts, exc
                )
                time.sleep(delay)
    assert last_exc is not None
    logger.error("%s failed after %d attempts", name, attempts)
    raise last_exc


def run_extract_post_info(image_path: Path) -> Dict[str, Any]:
    cmd = ["node", str(EXTRACT_SCRIPT), str(image_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"extract_post_info failed: {message}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON from extract_post_info: {exc}\n{result.stdout}"
        ) from exc


def kill_switch_guard(state: AgentState, kill_switch: KillSwitch) -> AgentState:
    if kill_switch.triggered():
        if not state.get("terminated"):
            logger.warning("Kill switch active; flagging termination")
        state["terminated"] = True
    return state


def build_agent_graph(
    orchestrator: DeepSeekR1Orchestrator,
    kill_switch: KillSwitch,
    continue_waiter: ContinueWaiter,
    *,
    use_perplexity: bool,
) -> StateGraph:
    graph = StateGraph(AgentState)

    def plan_node(state: AgentState) -> AgentState:
        state = kill_switch_guard(state, kill_switch)
        if state.get("terminated"):
            return state
        if state.get("plan") and (
            time.time() - state.get("last_plan_refresh", 0) < 300
        ):
            return state
        logger.info("Generating plan for objective: %s", state["objective"])
        plan = orchestrator.generate_plan(state["objective"])
        state["plan"] = plan
        state["last_plan_refresh"] = time.time()
        logger.info("Plan ready: %s", plan)
        state.setdefault("history", []).append(f"Plan ready: {plan}")
        return state

    def fetch_post_node(state: AgentState) -> AgentState:
        state = kill_switch_guard(state, kill_switch)
        if state.get("terminated"):
            return state
        logger.info("Fetching next Reddit post via capture_next_post")
        payload_raw = with_retries(capture_next_post)
        if not payload_raw:
            raise RuntimeError("capture_next_post returned no post")
        payload = to_json_safe(payload_raw)
        state["post_payload"] = payload
        state["skip_reason"] = None
        image_name = (
            Path(payload["image_path"]).name if "image_path" in payload else "unknown"
        )
        logger.info("Captured post image: %s", image_name)
        state.setdefault("history", []).append(
            f"Fetched post: {Path(payload['image_path']).name if 'image_path' in payload else 'unknown'}"
        )
        return state

    def extract_node(state: AgentState) -> AgentState:
        state = kill_switch_guard(state, kill_switch)
        if state.get("terminated"):
            return state
        image_path = Path(state["post_payload"]["image_path"])  # type: ignore[index]
        analysis = with_retries(run_extract_post_info, image_path)
        analysis = to_json_safe(analysis)
        state["post_analysis"] = analysis
        classification = "ad" if analysis.get("isAd") else "organic"
        language = analysis.get("language", "unknown")
        logger.info(
            "Post analysis complete | classification=%s language=%s isArabic=%s",
            classification,
            language,
            analysis.get("isArabic"),
        )
        state.setdefault("history", []).append(
            f"Post classified: {classification} (lang={language})"
        )
        if analysis.get("isAd"):
            state["skip_reason"] = "ad_detected"
            state.setdefault("history", []).append("Skipping advertisement")
            logger.info("Skipping post because it is an advertisement")
        elif analysis.get("isArabic"):
            state["skip_reason"] = "arabic_post"
            state.setdefault("history", []).append("Skipping Arabic-language post")
            logger.info("Skipping post because it is detected as Arabic content")
        return state

    def decide_node(state: AgentState) -> AgentState:
        state = kill_switch_guard(state, kill_switch)
        if state.get("terminated"):
            return state
        summary = to_json_safe(
            {
                "post": state.get("post_payload"),
                "analysis": state.get("post_analysis"),
                "objective": state["objective"],
            }
        )
        decision = orchestrator.decide_action(state["objective"], summary)
        state["decision"] = decision  # type: ignore[assignment]
        logger.info("Model decision: %s", decision)
        state.setdefault("history", []).append(f"Decided action: {decision}")
        return state

    def act_node(state: AgentState) -> AgentState:
        state = kill_switch_guard(state, kill_switch)
        if state.get("terminated"):
            return state
        action = state.get("decision")
        if action == "upvote":
            logger.info("Executing upvote action")
            with_retries(upvote_post)
        elif action == "downvote":
            logger.info("Executing downvote action")
            with_retries(downvote_post)
        elif action == "comment":
            logger.info("Opening comment composer")
            with_retries(comment_on_post)
        state.setdefault("history", []).append(f"Executed action: {action}")
        return state

    def describe_node(state: AgentState) -> AgentState:
        state = kill_switch_guard(state, kill_switch)
        if state.get("terminated"):
            return state
        if state.get("decision") != "comment":
            return state
        logger.info("Capturing detailed post view for description")
        detail_image = with_retries(capture_postcard_detail)
        state["last_detail_image"] = str(detail_image)

        if use_perplexity:
            explanation = with_retries(
                describe_post_image,
                detail_image,
                "Provide a clear, empathetic explanation of this Reddit post so I can respond helpfully.",
            )
            logger.info(
                "Received Perplexity explanation (truncated): %s", explanation[:200]
            )
        else:
            logger.info("Using local model description pipeline (skipping Perplexity)")
            explanation = orchestrator.summarize_post(
                state.get("post_analysis") or {},
                state.get("post_payload") or {},
                detail_image,
            )
            logger.info("Received local explanation (truncated): %s", explanation[:200])
        state["description"] = explanation
        state.setdefault("history", []).append("Captured and described post detail")
        return state

    def compose_node(state: AgentState) -> AgentState:
        state = kill_switch_guard(state, kill_switch)
        if state.get("terminated"):
            return state
        if state.get("decision") != "comment":
            return state
        explanation = state.get("description") or ""
        comment_text = orchestrator.compose_comment(state["objective"], explanation)
        logger.info("Composed comment (%d characters)", len(comment_text))
        state["composed_comment"] = comment_text
        state.setdefault("history", []).append("Composed comment text")
        return state

    def submit_node(state: AgentState) -> AgentState:
        state = kill_switch_guard(state, kill_switch)
        if state.get("terminated"):
            return state
        if state.get("decision") != "comment":
            return state
        comment_text = state.get("composed_comment")
        if not comment_text:
            raise RuntimeError("No comment composed for submission")
        logger.info("Pasting composed comment into Reddit composer")
        try:
            with_retries(write_reddit_comment, comment_text, auto_submit=False)
        except Exception as exc:
            logger.warning(
                "Comment workflow failed; returning to feed to retry: %s", exc
            )
            navigate_back_to_feed("comment validation failure")
            state.setdefault("history", []).append("Comment workflow failed; retrying")
            state["skip_reason"] = "comment_failed"
            state["decision"] = None
            state["description"] = None
            state["composed_comment"] = None
            state["last_detail_image"] = None
            return state

        logger.info(
            "Comment pasted; press 'c' after submitting or cancelling to continue"
        )
        proceeded = continue_waiter.wait(kill_switch)
        if not proceeded:
            state.setdefault("history", []).append("Comment review interrupted")
            state["terminated"] = True
            return state
        logger.info("User confirmed continue; resuming navigation")
        state.setdefault("history", []).append("Comment handled manually")
        return state

    def navigate_back_node(state: AgentState) -> AgentState:
        state = kill_switch_guard(state, kill_switch)
        if state.get("terminated"):
            return state
        if state.get("decision") == "comment":
            navigate_back_to_feed("continuing loop")
            state.setdefault("history", []).append("Navigated back to feed")
        else:
            if state.get("skip_reason") == "comment_failed":
                logger.info("Navigation already handled after comment failure")
            else:
                logger.info("No navigation required at this stage")
        return state

    def loop_node(state: AgentState) -> AgentState:
        state = kill_switch_guard(state, kill_switch)
        state["loop_count"] = state.get("loop_count", 0) + 1
        max_loops = state.get("max_loops", 1)
        logger.info("Completed loop %d of %d", state["loop_count"], max_loops)
        state.setdefault("history", []).append(f"Loop {state['loop_count']} completed")
        if kill_switch.triggered() or state["loop_count"] >= max_loops:
            logger.info("Termination conditions met; stopping agent")
            state["terminated"] = True
        state["post_payload"] = {}
        state["post_analysis"] = {}
        state["decision"] = None
        state["description"] = None
        state["composed_comment"] = None
        state["skip_reason"] = None
        return state

    graph.add_node("plan", plan_node)
    graph.add_node("fetch_post", fetch_post_node)
    graph.add_node("extract", extract_node)
    graph.add_node("decide", decide_node)
    graph.add_node("act", act_node)
    graph.add_node("describe", describe_node)
    graph.add_node("compose", compose_node)
    graph.add_node("submit", submit_node)
    graph.add_node("navigate_back", navigate_back_node)
    graph.add_node("loop", loop_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "fetch_post")
    graph.add_edge("fetch_post", "extract")

    graph.add_conditional_edges(
        "extract",
        lambda state: "skip" if state.get("skip_reason") else "continue",
        {"skip": "fetch_post", "continue": "decide"},
    )

    graph.add_edge("decide", "act")

    graph.add_conditional_edges(
        "act",
        lambda state: "comment" if state.get("decision") == "comment" else "no_comment",
        {"comment": "describe", "no_comment": "navigate_back"},
    )

    graph.add_edge("describe", "compose")
    graph.add_edge("compose", "submit")
    graph.add_edge("submit", "navigate_back")

    graph.add_edge("navigate_back", "loop")

    graph.add_conditional_edges(
        "loop",
        lambda state: "end" if state.get("terminated") else "continue",
        {"end": END, "continue": "fetch_post"},
    )

    return graph


def run_agent(
    objective: str,
    *,
    max_loops: int = 3,
    use_perplexity: bool = True,
) -> AgentState:
    orchestrator = DeepSeekR1Orchestrator()
    kill_switch = KillSwitch()
    continue_waiter = ContinueWaiter()
    graph = build_agent_graph(
        orchestrator,
        kill_switch,
        continue_waiter,
        use_perplexity=use_perplexity,
    )
    app = graph.compile(checkpointer=MemorySaver())

    initial_state: AgentState = {
        "objective": objective,
        "plan": [],
        "last_plan_refresh": 0.0,
        "current_step_index": 0,
        "loop_count": 0,
        "max_loops": max_loops,
        "history": [],
        "terminated": False,
    }

    logger.info(
        "Starting Reddit agent | objective='%s' max_loops=%d use_perplexity=%s",
        objective,
        max_loops,
        use_perplexity,
    )
    kill_switch.start()
    thread_id = f"reddit-agent-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        final_state = app.invoke(initial_state, config=config)
    finally:
        kill_switch.stop()
        continue_waiter.stop()
    logger.info("Agent run complete")
    return final_state


def main(argv: Sequence[str]) -> int:
    if len(argv) < 1:
        print(
            "Usage: python reddit_langgraph_agent.py '<objective>' [max_loops] [perplexity|local]",
            file=sys.stderr,
        )
        return 1
    objective = argv[0]
    max_loops = int(argv[1]) if len(argv) > 1 else 3
    if len(argv) > 2:
        describe_mode = argv[2].strip().lower()
    else:
        describe_mode = os.getenv("REDDIT_AGENT_DESCRIBE_MODE", "perplexity").lower()
    use_perplexity = describe_mode not in {"local", "deepseek", "offline"}
    configure_logging()
    final_state = run_agent(
        objective,
        max_loops=max_loops,
        use_perplexity=use_perplexity,
    )
    print(
        json.dumps({k: v for k, v in final_state.items() if k != "objective"}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
