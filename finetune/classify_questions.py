#!/usr/bin/env python3
"""Extract Somali speaker turns, classify questions with Jev, and pair replies."""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

API_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "~typesafe/jev-latest"


def compact(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def response_order(path):
    match = re.search(r"(?:window|single)_(\d+)", path.name)
    return int(match.group(1)) if match else 10**9


def load_offsets(recording_dir):
    manifest = recording_dir / "transcripts.diarized.json"
    if not manifest.exists():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        item.get("response_file"): float(item.get("offset_seconds", 0.0))
        for item in data.get("windows", [])
        if item.get("response_file")
    }


def merge_segments(segments, offset, source_file, recording):
    turns = []
    for seg in segments:
        text = compact(seg.get("text"))
        if not text:
            continue
        speaker = seg.get("speaker")
        start = offset + float(seg.get("start", 0.0))
        end = offset + float(seg.get("end", seg.get("start", 0.0)))
        if (turns and turns[-1]["speaker"] == speaker
                and start - turns[-1]["end"] <= 1.5):
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = end
        else:
            turns.append({
                "recording": recording,
                "source_file": source_file,
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            })
    return turns


def extract_turns(root):
    by_recording = defaultdict(list)
    for diarized in root.glob("*/*/diarized"):
        recording = str(diarized.parent.relative_to(root))
        offsets = load_offsets(diarized)
        for path in sorted(diarized.glob("*.response.json"), key=response_order):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"warning: skipping {path}: {exc}", file=sys.stderr)
                continue
            offset = offsets.get(path.name, 0.0)
            by_recording[recording].extend(merge_segments(
                payload.get("segments", []), offset,
                str(path.relative_to(root)), recording,
            ))

    # Overlapping ASR windows can repeat the same turn. Keep the earliest copy.
    output = []
    for recording, turns in by_recording.items():
        turns.sort(key=lambda x: (x["start"], x["end"]))
        accepted = []
        for turn in turns:
            norm = re.sub(r"[^\w]+", " ", turn["text"].lower()).strip()
            duplicate_at = None
            for pos in range(len(accepted) - 1, max(-1, len(accepted) - 12), -1):
                prior = accepted[pos]
                if turn["start"] - prior["end"] > 12:
                    break
                if turn["speaker"] != prior["speaker"]:
                    continue
                prior_norm = re.sub(r"[^\w]+", " ", prior["text"].lower()).strip()
                overlap = max(0.0, min(turn["end"], prior["end"]) -
                              max(turn["start"], prior["start"]))
                shorter = max(0.001, min(turn["end"] - turn["start"],
                                         prior["end"] - prior["start"]))
                same_window_copy = overlap / shorter >= 0.75
                text_copy = norm == prior_norm or norm in prior_norm or prior_norm in norm
                if same_window_copy or text_copy:
                    duplicate_at = pos
                    break
            if duplicate_at is not None:
                if len(turn["text"]) > len(accepted[duplicate_at]["text"]):
                    accepted[duplicate_at] = turn
                continue
            accepted.append(turn)
        accepted.sort(key=lambda x: (x["start"], x["end"]))
        for index, turn in enumerate(accepted):
            turn["start"] = round(turn["start"], 3)
            turn["end"] = round(turn["end"], 3)
            turn["id"] = f"{recording}:{index:06d}"
        output.extend(accepted)
    return output


def jev_request(api_key, model, batch, timeout, retries):
    state = [{"id": t["id"], "speaker": t["speaker"], "text": t["text"]}
             for t in batch]
    questions = {}
    for index, turn in enumerate(batch):
        questions[f"q{index}"] = {
            "type": "noul",
            "instructions": (
                f"Is the utterance whose id is {turn['id']!r} a question? "
                "Judge communicative intent in Somali, even when ASR punctuation is missing "
                "or transcription is imperfect. Rhetorical and clarification questions count. "
                "Commands, greetings, and statements do not count."
            ),
            "criteria": {
                "true": "The speaker is asking for information, confirmation, or an answer.",
                "false": "The utterance is not a question."
            },
        }
    body = json.dumps({
        "model": model,
        "state": state,
        "questions": questions,
    }).encode("utf-8")
    request = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Somali question-pair extraction",
    })
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:1000]
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        time.sleep(min(2 ** attempt, 30))


def probability(answer):
    if not isinstance(answer, dict):
        raise ValueError(f"unexpected Jev answer: {answer!r}")
    value = answer.get("noul")
    if value is None:
        value = answer.get("probability")
    if value is None:
        raise ValueError(f"missing noul probability: {answer!r}")
    return float(value)


def read_labels(path):
    labels = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    labels[row["id"]] = row
                except (json.JSONDecodeError, KeyError):
                    pass
    return labels


def write_jsonl(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def make_pairs(turns, labels, threshold, max_reply_gap):
    grouped = defaultdict(list)
    for turn in turns:
        grouped[turn["recording"]].append(turn)
    questions, pairs = [], []
    for recording_turns in grouped.values():
        recording_turns.sort(key=lambda x: x["start"])
        for index, turn in enumerate(recording_turns):
            label = labels.get(turn["id"])
            if not label or label["question_probability"] < threshold:
                continue
            question = {**turn, **label}
            questions.append(question)
            for reply in recording_turns[index + 1:]:
                if reply["start"] - turn["end"] > max_reply_gap:
                    break
                if reply["speaker"] != turn["speaker"]:
                    pairs.append({
                        "id": turn["id"],
                        "recording": turn["recording"],
                        "question": turn["text"],
                        "question_speaker": turn["speaker"],
                        "question_start": turn["start"],
                        "question_end": turn["end"],
                        "question_probability": label["question_probability"],
                        "answer": reply["text"],
                        "answer_speaker": reply["speaker"],
                        "answer_start": reply["start"],
                        "answer_end": reply["end"],
                        "model": label["model"],
                    })
                    break
    return questions, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("question_classifier_output"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--max-reply-gap", type=float, default=45.0)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    root = args.root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    turns = extract_turns(root)
    if args.max_turns is not None:
        turns = turns[:args.max_turns]
    print(f"extracted {len(turns):,} unique speaker turns")
    if args.dry_run:
        for turn in turns[:5]:
            print(json.dumps(turn, ensure_ascii=False))
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    labels_path = args.output_dir / "turn_labels.jsonl"
    labels = read_labels(labels_path)
    pending = [turn for turn in turns if turn["id"] not in labels]
    print(f"resuming with {len(labels):,} labels; {len(pending):,} pending")
    with labels_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            result = jev_request(api_key, args.model, batch, args.timeout, args.retries)
            answers = result.get("answers", {})
            resolved_model = result.get("model", args.model)
            for index, turn in enumerate(batch):
                row = {
                    "id": turn["id"],
                    "question_probability": probability(answers.get(f"q{index}")),
                    "model": resolved_model,
                }
                labels[turn["id"]] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            done = min(start + len(batch), len(pending))
            print(f"classified {done:,}/{len(pending):,} pending", flush=True)

    questions, pairs = make_pairs(turns, labels, args.threshold, args.max_reply_gap)
    write_jsonl(args.output_dir / "questions.jsonl", questions)
    write_jsonl(args.output_dir / "question_answer_pairs.jsonl", pairs)
    summary = {
        "turns": len(turns), "questions": len(questions), "pairs": len(pairs),
        "threshold": args.threshold, "requested_model": args.model,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
