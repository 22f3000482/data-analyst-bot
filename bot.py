"""
Data-Analyst Telegram Bot
=========================
- Long-polls Telegram for messages
- Feeds each message to an LLM (OpenAI gpt-4o) with a run_python tool
- Replies with exactly one JSON object: {"answer": ..., "log_url": "..."}
- Logs every step to run.jsonl, served publicly at /run.jsonl
- Self-pings /health every 10 min to stay awake on free hosts
"""

import os
import io
import sys
import json
import time
import threading
import traceback
import contextlib
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config (all from environment variables - set these in Render, never in code)
# ---------------------------------------------------------------------------
BOT_TOKEN = os.environ["BOT_TOKEN"]                 # from BotFather
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]       # from platform.openai.com
BASE_URL = os.environ["BASE_URL"]                   # e.g. https://yourapp.onrender.com
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
LOG_FILE = "run.jsonl"
MAX_TOOL_STEPS = 10
WALL_CLOCK_BUDGET_SECONDS = 210  # stay well under the 300s grader timeout

client = OpenAI(api_key=OPENAI_API_KEY)

# Per-chat conversation history: {chat_id: [ {role, content}, ... ]}
chat_histories = {}
HISTORY_TURNS_KEPT = 20

# ---------------------------------------------------------------------------
# Logging - JSONL, one line per event, served publicly at /run.jsonl
# ---------------------------------------------------------------------------
log_lock = threading.Lock()


def log_event(event: dict):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")


# ---------------------------------------------------------------------------
# The run_python tool - executes model-written code, captures stdout
# ---------------------------------------------------------------------------
def run_python_tool(code: str) -> str:
    """Executes `code` and returns captured stdout (or the error trace)."""
    buf = io.StringIO()
    try:
        safe_globals = {"__name__": "__main__"}
        with contextlib.redirect_stdout(buf):
            exec(code, safe_globals)
        output = buf.getvalue()
    except Exception:
        output = "ERROR:\n" + traceback.format_exc()
    # cap output length so we don't blow up the context window
    if len(output) > 8000:
        output = output[:8000] + "\n...[truncated]"
    return output


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code on the server and return whatever it prints "
                "to stdout. Use this to download datasets (requests), parse "
                "them (pandas, BeautifulSoup, openpyxl) and compute the answer. "
                "Always print() the values you need to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are a data-analyst agent replying inside a Telegram chat.

Rules:
- Answer the LATEST user message. Earlier messages in this chat are context (some tasks are multi-turn).
- Use the run_python tool to fetch and compute anything you can (pandas, requests, BeautifulSoup, openpyxl are installed). Never guess a number you could compute.
- If a message is only setup (e.g. "I will send data next"), still reply with a small JSON acknowledgement - do not stay silent.
- Your FINAL reply must be ONLY a single JSON object and NOTHING else - no markdown fences, no explanation text before or after.
- The JSON must have exactly the shape the question asks for, plus always include a "log_url" key. Use the literal placeholder string "LOG_URL_PLACEHOLDER" for log_url - the calling code will replace it with the real URL.
- Never add extra keys beyond what the question's requested shape needs (plus log_url).
- If you cannot finish in time, still answer with your best guess rather than nothing.
"""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def run_agent(chat_id: int, user_message: str) -> str:
    deadline = time.time() + WALL_CLOCK_BUDGET_SECONDS

    history = chat_histories.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_message})
    # keep only the last N turns to bound context size
    history[:] = history[-HISTORY_TURNS_KEPT:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    final_text = None
    for step in range(MAX_TOOL_STEPS):
        time_left = deadline - time.time()
        force_final = time_left < 20  # stop calling tools if we're almost out of time

        kwargs = dict(model=MODEL_NAME, messages=messages)
        if not force_final:
            kwargs["tools"] = TOOLS

        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls and not force_final:
            # record the assistant's tool-call message
            messages.append(msg.model_dump(exclude_none=True))
            for tc in tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                code = args.get("code", "")
                log_event({"chat_id": chat_id, "type": "tool_call", "code": code})
                result = run_python_tool(code)
                log_event({"chat_id": chat_id, "type": "tool_result", "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
            continue  # loop again with the tool result in context

        # No tool call (or forced final) -> this is the final answer text
        final_text = msg.content or ""
        break

    if final_text is None:
        final_text = '{"answer": "no answer produced", "log_url": "LOG_URL_PLACEHOLDER"}'

    history.append({"role": "assistant", "content": final_text})
    history[:] = history[-HISTORY_TURNS_KEPT:]

    log_event({"chat_id": chat_id, "type": "final_raw", "text": final_text})
    return final_text


# ---------------------------------------------------------------------------
# Robust JSON extraction from model output
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    text = text.strip()
    # strip markdown fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    # find first balanced {...}
    start = text.find("{")
    if start == -1:
        return {"answer": text}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    return {"answer": text}
                if not isinstance(parsed, dict) or "answer" not in parsed:
                    parsed = {"answer": parsed}
                return parsed
    return {"answer": text}


def build_reply(chat_id: int, user_message: str) -> str:
    try:
        raw = run_agent(chat_id, user_message)
        parsed = extract_json(raw)
    except Exception:
        log_event({"chat_id": chat_id, "type": "error", "trace": traceback.format_exc()})
        parsed = {"answer": "internal error"}

    parsed["log_url"] = f"{BASE_URL}/run.jsonl"
    return json.dumps(parsed)


# ---------------------------------------------------------------------------
# Telegram long-polling loop
# ---------------------------------------------------------------------------
def send_message(chat_id: int, text: str):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=30,
    )


def telegram_poll_loop():
    offset = None
    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            resp = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=40)
            data = resp.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message")
                if not message or "text" not in message:
                    continue
                chat_id = message["chat"]["id"]
                text = message["text"]
                log_event({"chat_id": chat_id, "type": "incoming", "text": text})
                reply = build_reply(chat_id, text)
                log_event({"chat_id": chat_id, "type": "outgoing", "text": reply})
                send_message(chat_id, reply)
        except Exception:
            log_event({"type": "poll_error", "trace": traceback.format_exc()})
            time.sleep(3)


def self_ping_loop():
    while True:
        time.sleep(600)  # every 10 minutes
        try:
            requests.get(f"{BASE_URL}/health", timeout=20)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/run.jsonl", response_class=PlainTextResponse)
def get_log():
    if not os.path.exists(LOG_FILE):
        return ""
    with open(LOG_FILE) as f:
        return f.read()


@app.on_event("startup")
def startup():
    threading.Thread(target=telegram_poll_loop, daemon=True).start()
    threading.Thread(target=self_ping_loop, daemon=True).start()
