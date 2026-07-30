# Data-Analyst Telegram Bot

A Telegram bot that answers data-analysis questions using an LLM agent with
a `run_python` tool, and replies with a single JSON object:

```json
{"answer": <shaped as the question asks>, "log_url": "https://your-host/run.jsonl"}
```

## Environment variables (set these on your host, never in code)

| Variable        | Description                                      |
|-----------------|---------------------------------------------------|
| `BOT_TOKEN`     | Telegram bot token from @BotFather                 |
| `OPENAI_API_KEY`| OpenAI API key from platform.openai.com            |
| `BASE_URL`      | Public URL of this deployed service, no trailing slash (e.g. `https://yourapp.onrender.com`) |
| `MODEL_NAME`    | Optional, defaults to `gpt-4o`                     |

## Run locally

```bash
pip install -r requirements.txt
export BOT_TOKEN=...
export OPENAI_API_KEY=...
export BASE_URL=http://localhost:8000
uvicorn bot:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health` -> `{"ok": true}`
- `GET /run.jsonl` -> plain-text JSONL log of every run
