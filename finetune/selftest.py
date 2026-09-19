"""Checks that GLMFormat builds exactly the ids the demo gets by tokenizing the prompt string.

Run before training; exits non-zero on mismatch.
"""
import sys

from common import LLM_PATH, SPEECH_SYSTEM, TEXT_SYSTEM, GLMFormat, log

L = log("selftest")


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(LLM_PATH), trust_remote_code=True)
    g = GLMFormat(tok)
    # The tokenizer only has string tokens <|audio_0|>..<|audio_16382|>; codebook id 16383 exists only as
    # embedding row audio_offset+16383. So compare against string tokenization inside that range, and
    # build ids arithmetically (as the demo decodes them) everywhere.
    audio = [5, 16382, 0, 42]
    audio_str = "<|begin_of_audio|>" + "".join(f"<|audio_{x}|>" for x in audio) + "<|end_of_audio|>"
    cases = [
        (g.prompt(SPEECH_SYSTEM, g.audio(audio)),
         f"<|system|>\n{SPEECH_SYSTEM}<|user|>\n{audio_str}<|assistant|>streaming_transcription\n"),
        (g.prompt(TEXT_SYSTEM, g.enc("Iska warran, sidee tahay?")),
         f"<|system|>\n{TEXT_SYSTEM}<|user|>\nIska warran, sidee tahay?<|assistant|>streaming_transcription\n"),
    ]
    ok = True
    for built, s in cases:
        ref = tok([s])["input_ids"][0]
        if built != ref:
            ok = False
            L.error("MISMATCH\n built=%s\n ref  =%s", built[:40], ref[:40])
    inter = g.interleave(list(range(30)), [1] * 60)
    ok &= inter[:13] == list(range(13)) and len(inter) == 90
    import json
    vocab = json.loads((LLM_PATH / "config.json").read_text())["padded_vocab_size"]
    ok &= g.audio([16383])[1] == g.audio_offset + 16383 < vocab     # every codebook id has an embedding row
    L.info("user token id %d, audio offset %d, prefix %s", g.user_id, g.audio_offset, g.prefix)
    L.info("selftest %s", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
