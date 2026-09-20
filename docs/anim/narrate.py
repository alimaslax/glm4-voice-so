"""Turn narration.py into one narration track per scene, via OpenRouter."""

import base64
import json
import os
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from narration import NARRATION  # noqa: E402

MODEL = os.environ.get("NARRATE_MODEL", "openai/gpt-audio-mini")
VOICE = os.environ.get("NARRATE_VOICE", "ash")
KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media", "narration")

PROMPT = (
    "You are a narrator for a technical explainer video. Read the text between "
    "the markers aloud, verbatim: no greetings, no commentary, no additions or "
    "omissions. Calm, clear, unhurried documentary delivery.\n\n"
    "<<<TEXT>>>\n{text}\n<<<END>>>"
)


def speak(text):
    body = {
        "model": MODEL,
        "modalities": ["text", "audio"],
        "audio": {"voice": VOICE, "format": "pcm16"},
        "stream": True,
        "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
    )
    pcm, said = [], []
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            for c in d.get("choices", []):
                a = (c.get("delta") or {}).get("audio") or {}
                if a.get("data"):
                    pcm.append(base64.b64decode(a["data"]))
                if a.get("transcript"):
                    said.append(a["transcript"])
    return b"".join(pcm), "".join(said)


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def main():
    if not KEY:
        sys.exit("no OPENROUTER_API_KEY / OPEN_ROUTER in the environment")
    os.makedirs(OUT, exist_ok=True)
    for scene, text in NARRATION:
        want = " ".join(text.split())
        for attempt in (1, 2, 3):
            pcm, said = speak(want)
            # the model occasionally ad-libs; retry until it reads what we wrote
            ok = words(said) == words(want)
            if ok or attempt == 3:
                break
            print(f"   {scene}: retry {attempt} (transcript drifted)")
        secs = len(pcm) / 2 / 24000
        # AAC, not wav: these are committed next to the video, and the voice
        # costs an API call to regenerate
        dst = os.path.join(OUT, scene + ".m4a")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
             "-i", "pipe:0", "-ar", "48000", "-c:a", "aac", "-b:a", "96k", dst],
            input=pcm, check=True,
        )
        print(f"==> {dst}  {secs:.1f}s  verbatim={ok}")
        if not ok:
            print("    said:", " ".join(said.split())[:160])


if __name__ == "__main__":
    main()
