# Somali question-pair extraction with Jev

The classifier reads ElevenLabs Scribe diarized transcripts, merges adjacent
same-speaker segments, removes obvious overlap duplicates, asks OpenRouter Jev for
a calibrated question probability, and pairs accepted questions with the next
turn from a different speaker.

The transcript corpus remains at `/Users/mali/hf/so-work/scribe`; it is not copied
into this repository.

## Run

From the repository root:

```bash
set -a
source .env
set +a
export OPENROUTER_API_KEY="$OPEN_ROUTER"

python3 finetune/classify_questions.py \
  --root /Users/mali/hf/so-work/scribe \
  --output-dir outputs/jev_question_classifier \
  --max-turns 500
```

Remove `--max-turns 500` for the complete corpus. The pipeline resumes from
`turn_labels.jsonl`, so interrupted runs can be restarted safely.

Important options:

- `--threshold 0.70`: minimum probability retained as a question.
- `--batch-size 24`: turns classified per OpenRouter request.
- `--max-reply-gap 45`: maximum seconds between a question and paired reply.
- `--model '~typesafe/jev-latest'`: prototype alias; pin the concrete model after evaluation.

Outputs:

- `turn_labels.jsonl`: resumable Jev scores for every processed turn.
- `questions.jsonl`: question turns above the configured threshold.
- `question_answer_pairs.jsonl`: each question with the next other-speaker turn.
- `summary.json`: run totals and configuration.

The first 100-turn pilot resolved to `typesafe/jev-1.13-20260917` and produced
17 questions and 16 question-answer pairs at the 0.70 threshold. Jev appears
suitable for bulk mining, but a hand-labeled Somali evaluation should be used to
tune the threshold and compare recall against Gemini.
