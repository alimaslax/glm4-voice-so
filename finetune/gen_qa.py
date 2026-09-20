"""Generate Somali question/answer text pairs with an LLM, for the dialogue (speech-in -> speech-out) round.

The user reads the questions aloud; CosyVoice-3 (Omar) speaks the answers.  This stage only produces
text -- recording and synthesis happen after.

  python gen_qa.py --n 1200 --out $SO_WORK/qa/qa.jsonl
  python gen_qa.py --n 1200 --out $SO_WORK/qa/qa.jsonl --sheet    # also write the reading script

Needs OPENROUTER_API_KEY (finetune/run.sh already passes .env into the container).
"""
import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import log

L = log("gen_qa")

API = "https://openrouter.ai/api/v1/chat/completions"

# The grid is the whole trick: diversity comes from walking these axes, not from a high temperature,
# so a thousand calls don't collapse onto the same half-dozen questions.
TOPICS = [
    "cunto iyo karinta", "safar iyo dalxiis", "caafimaad iyo jimicsi", "waxbarasho iyo dugsi",
    "shaqo iyo ganacsi", "tignoolajiyad iyo taleefan", "ciyaaraha iyo kubbadda cagta",
    "qoys iyo carruur", "dhaqan iyo suugaan Soomaaliyeed", "cimilo iyo xilliyada",
    "gaadiid iyo socdaal magaalo", "lacag iyo kaydinta", "warbaahin iyo wararka",
    "xiriir iyo saaxiibtinimo", "guri iyo nadaafad", "xayawaan iyo deegaan",
    "taariikh iyo juqraafi", "diin iyo anshax", "farshaxan iyo muusik", "caadooyin arooska",
    "biyo iyo beeraha", "dhakhtar iyo dawooyin", "ciyaaraha carruurta", "luqad iyo barashadeeda",
]
REGISTERS = [
    "af maalmeed oo fudud", "si edeb leh oo rasmi ah", "qof saaxiibkaa ah oo kaa weydiinaya",
    "hab su'aal deg deg ah oo gaaban", "qof yar oo wax weydiinaya", "hab macallin oo wax caddaynaya",
]
FORMS = [
    "su'aal 'maxay' ah", "su'aal 'sidee' ah", "su'aal 'goorma' ah", "su'aal 'maxaa yeelay' ah",
    "su'aal 'xaggee' ah", "su'aal haa/maya ah", "codsi caawimaad ah", "su'aal is barbar dhig ah",
]

SYSTEM = (
    "You write natural spoken Somali (af-Soomaali) training data for a voice assistant. "
    "You output ONLY a JSON array. Every element is an object with exactly two keys: "
    '"q" (the question, as a person would SAY it out loud) and "a" (the assistant\'s spoken answer). '
    "Rules: both fields are Somali only -- no English, no Arabic script, no emoji, no markdown, no lists. "
    "Write numbers as words (tobon, not 10) because the text is read aloud. "
    "Questions are 4-14 words. Answers are 1-3 sentences, 12-45 words, and must sound like speech, "
    "not like an article: direct, warm, and finished -- never trailing off, never asking a question back. "
    "Use ordinary punctuation (. , ?) so the TTS paces correctly."
)


def call(model, key, prompt, temperature, retries=4):
    body = json.dumps({
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            wait = 2 ** attempt
            L.warning("call failed (%s), retry in %ds", e, wait)
            time.sleep(wait)
    return ""


def parse(text):
    """Models wrap the array in prose or a fence often enough that we cut to the outermost brackets."""
    i, j = text.find("["), text.rfind("]")
    if i < 0 or j < 0:
        return []
    try:
        rows = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("q") and r.get("a")]


NON_SOMALI = re.compile(r"[^\w\s.,?!'\-]", re.UNICODE)


def norm(s):
    return re.sub(r"\s+", " ", NON_SOMALI.sub("", s.lower())).strip()


def acceptable(q, a):
    """Cheap gates that catch the usual generation garbage before it costs recording or GPU time."""
    if NON_SOMALI.search(q) or NON_SOMALI.search(a):
        return "non_somali_chars"
    if not 3 <= len(q.split()) <= 18:
        return "q_length"
    if not 8 <= len(a.split()) <= 60:
        return "a_length"
    if a.rstrip().endswith("?"):
        return "a_asks_back"
    if re.search(r"[a-z]{3,}\d|http|www", q + a, re.I):
        return "junk"
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000, help="target number of accepted pairs")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="google/gemini-2.5-pro")
    p.add_argument("--per-call", type=int, default=12)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sheet", action="store_true", help="also write a numbered reading script next to --out")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY2")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen, kept, dropped = set(), [], {}
    # Resuming matters: a thousand pairs is many minutes of API time and the run gets interrupted.
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            seen.add(norm(r["question"]))
            kept.append(r)
        L.info("resuming with %d existing pairs", len(kept))

    with out.open("a") as f:
        while len(kept) < a.n:
            topic, reg, form = rng.choice(TOPICS), rng.choice(REGISTERS), rng.choice(FORMS)
            prompt = (f"Qor {a.per_call} lammaane su'aal-jawaab oo Soomaali ah.\n"
                      f"Mawduuca: {topic}\nHabka hadalka: {reg}\nNooca su'aasha: {form}\n"
                      f"Ha ku celcelin su'aalo isku mid ah. Soo celi JSON array oo kaliya.")
            rows = parse(call(a.model, key, prompt, a.temperature))
            if not rows:
                dropped["empty_batch"] = dropped.get("empty_batch", 0) + 1
                continue
            new = 0
            for r in rows:
                q, ans = " ".join(r["q"].split()), " ".join(r["a"].split())
                why = acceptable(q, ans)
                if why:
                    dropped[why] = dropped.get(why, 0) + 1
                    continue
                k = norm(q)
                if k in seen:
                    dropped["duplicate"] = dropped.get("duplicate", 0) + 1
                    continue
                seen.add(k)
                rec = {"id": f"qa{len(kept):06d}", "topic": topic, "register": reg, "form": form,
                       "question": q, "answer": ans}
                kept.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                new += 1
            f.flush()
            L.info("%d/%d  (+%d from %s)", len(kept), a.n, new, topic)

    L.info("wrote %d pairs -> %s", len(kept), out)
    L.info("dropped: %s", json.dumps(dropped, sort_keys=True))

    # The answers file feeds /workspace/cosyvoice/scripts/somali_answers.py verbatim.
    ans_path = out.with_name("answers.jsonl")
    with ans_path.open("w") as f:
        for r in kept:
            f.write(json.dumps({"id": r["id"], "text": r["answer"], "pace": "medium"}, ensure_ascii=False) + "\n")
    L.info("wrote %s", ans_path)

    if a.sheet:
        sheet = out.with_name("read_me.md")
        lines = ["# Su'aalaha la akhrinayo", "",
                 "Hal su'aal hal xariiq. Ka hor su'aal kasta, sug hal ilbiriqsi, ka dibna akhri.", ""]
        for i, r in enumerate(kept):
            if i % 50 == 0:
                lines += ["", f"## Bog {i // 50 + 1}  ({r['id']} …)", ""]
            lines.append(f"{i + 1:4d}. [{r['id']}]  {r['question']}")
        sheet.write_text("\n".join(lines) + "\n")
        L.info("wrote %s", sheet)


if __name__ == "__main__":
    main()
