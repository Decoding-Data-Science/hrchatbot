import os
import threading
import time
from typing import Callable

import requests


TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}" if TELEGRAM_TOKEN else None


def _send_message(chat_id: int, text: str) -> None:
    """Send a plain-text Telegram reply, splitting long answers safely."""
    if not TELEGRAM_API:
        return

    text = str(text or "I couldn't generate an answer.")
    for start in range(0, len(text), 4000):
        chunk = text[start:start + 4000]
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": chunk},
            timeout=20,
        ).raise_for_status()


def _send_typing(chat_id: int) -> None:
    if not TELEGRAM_API:
        return
    try:
        requests.post(
            f"{TELEGRAM_API}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"},
            timeout=10,
        )
    except Exception:
        pass


def _poll(answer_fn: Callable[[str, str | None], str]) -> None:
    """Long-poll Telegram and pass questions to the existing HR answer function."""
    if not TELEGRAM_API:
        print("[Telegram] TELEGRAM_BOT_TOKEN not set; Telegram integration disabled.")
        return

    print("[Telegram] Bot polling started.")
    offset = None

    while True:
        try:
            params = {"timeout": 45}
            if offset is not None:
                params["offset"] = offset

            response = requests.get(
                f"{TELEGRAM_API}/getUpdates",
                params=params,
                timeout=55,
            )
            response.raise_for_status()
            payload = response.json()

            for update in payload.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message") or update.get("edited_message")
                if not message:
                    continue

                chat = message.get("chat", {})
                chat_id = chat.get("id")
                text = (message.get("text") or "").strip()

                if not chat_id or not text:
                    continue

                if text.startswith("/start"):
                    _send_message(
                        chat_id,
                        "Welcome to DDS HR Enterprise. Ask me a question about the HR handbook, policies, benefits, guides, roles, or company information.",
                    )
                    continue

                if text.startswith("/help"):
                    _send_message(
                        chat_id,
                        "Just type an HR question, for example: 'How do I request time off?'",
                    )
                    continue

                if text.startswith("/"):
                    _send_message(chat_id, "Please type your HR question as a normal message.")
                    continue

                _send_typing(chat_id)

                try:
                    answer = answer_fn(text, None)
                    _send_message(chat_id, answer)
                except Exception as exc:
                    print(f"[Telegram] Error answering message: {exc}")
                    _send_message(
                        chat_id,
                        "Sorry, I couldn't process that question right now. Please try again.",
                    )

        except Exception as exc:
            print(f"[Telegram] Polling error: {exc}")
            time.sleep(5)


def start_telegram_bot(answer_fn: Callable[[str, str | None], str]):
    """Start Telegram polling in a background thread so Gradio can keep running."""
    if not TELEGRAM_TOKEN:
        print("[Telegram] TELEGRAM_BOT_TOKEN not set; skipping bot startup.")
        return None

    thread = threading.Thread(target=_poll, args=(answer_fn,), daemon=True)
    thread.start()
    return thread
