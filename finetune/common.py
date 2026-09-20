"""Shared paths, prompt format and helpers for the Somali / Omar fine-tuning pipeline.

Every stage reads its locations from environment variables (set by finetune/run.sh),
so the same code runs unchanged on any VM with the network volume mounted.
"""
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "cosyvoice"))
sys.path.insert(0, str(REPO / "third_party" / "Matcha-TTS"))

DATA = Path(os.environ.get("SO_DATA", "/workspace/so-data"))    # raw downloads (buckets)
WORK = Path(os.environ.get("SO_WORK", "/workspace/so-train"))   # everything we generate
MODELS = Path(os.environ.get("SO_MODELS", "/workspace/glm-4-voice/models"))

TRANSCRIPTS_DIR = Path(os.environ.get("SO_TRANSCRIPTS", DATA / "transcripts"))   # bucket mai/ (or scribe/ relabel)
SOMALI = WORK / os.environ.get("SO_SOMALI", "somali")   # Track A manifests/tokens/sft; e.g. somali_scribe for the relabel
ASR_DIR = WORK / os.environ.get("SO_ASR", "asr")        # MMS datasets; e.g. asr_scribe3 for the Scribe relabel
PROCESSED_DIR = Path(os.environ.get("SO_PROCESSED", DATA / "processed"))   # lewenberg/so-duplex-processed (subset)
OMAR_DIR = PROCESSED_DIR / "omar"

LLM_PATH = MODELS / "glm-4-voice-9b"
TOKENIZER_PATH = MODELS / "glm-4-voice-tokenizer"
DECODER_PATH = MODELS / "glm-4-voice-decoder"

# --- GLM-4-Voice chat format (must match web_demo.py / orb_demo.py) -------------------
SPEECH_SYSTEM = ("User will provide you with a speech instruction. Do it step by step. First, think about the "
                 "instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. ")
TEXT_SYSTEM = ("User will provide you with a text instruction. Do it step by step. First, think about the "
               "instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens.")
ASR_SYSTEM = "User will provide you with a speech in Somali. Transcribe it into Somali text."
TTS_INSTRUCTION = "Si cod ah u akhri qoraalkan: "   # "Read this text aloud: "
TEXT_CHUNK, AUDIO_CHUNK = 13, 26
AUDIO_TOKEN_HZ = 12.5

# --- Omar mel features: CosyVoice-300M feat_extractor settings used by the GLM decoder -
MEL = dict(n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False)


def log(name):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def split_of(key, val=0.03, test=0.02):
    """Deterministic train/val/test split by a stable hash of the key (episode or video)."""
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def allow_trainer_resume():
    """torch>=2.6 loads with weights_only=True, but transformers 4.44 pickles the RNG state (numpy arrays)
    into every checkpoint -> resume fails. Allowlist exactly those numpy types (our own checkpoints)."""
    import numpy as np
    import torch
    torch.serialization.add_safe_globals(
        [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
        + [type(np.dtype(t)) for t in ("u1", "u4", "i4", "i8", "f4", "f8", "bool")])


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)   # atomic: a half-written file never looks finished


def load_audio(path, start=None, end=None):
    """Mono float32 numpy + sample rate. Reads only [start, end) seconds when given."""
    import soundfile as sf
    info = sf.info(str(path))
    sr = info.samplerate
    a = int(start * sr) if start is not None else 0
    b = int(end * sr) if end is not None else -1
    data, sr = sf.read(str(path), start=a, stop=None if b < 0 else b, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def load_speech_tokenizer():
    from transformers import WhisperFeatureExtractor
    from speech_tokenizer.modeling_whisper import WhisperVQEncoder
    model = WhisperVQEncoder.from_pretrained(str(TOKENIZER_PATH)).eval().cuda()   # same as the demos
    fe = WhisperFeatureExtractor.from_pretrained(str(TOKENIZER_PATH))
    return model, fe


def speech_tokens(model, fe, clips):
    """clips: list of (np.float32 mono, sr) -> list[list[int]] of 12.5 Hz audio token ids."""
    import torch
    from speech_tokenizer.utils import extract_speech_token
    return extract_speech_token(model, fe, [(torch.from_numpy(x).unsqueeze(0), sr) for x, sr in clips])


class GLMFormat:
    """Builds token-id sequences in GLM-4-Voice's chat format without string round-trips."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        c = tokenizer.convert_tokens_to_ids
        self.audio_offset = c("<|audio_0|>")
        self.boa, self.eoa = c("<|begin_of_audio|>"), c("<|end_of_audio|>")
        self.user_id = c("<|user|>")
        self.prefix = tokenizer.get_prefix_tokens()            # [gMASK] <sop>

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def audio(self, toks):
        return [self.boa] + [self.audio_offset + t for t in toks] + [self.eoa]

    def prompt(self, system, user_ids, streaming=True):
        head = "<|assistant|>streaming_transcription\n" if streaming else "<|assistant|>\n"
        return (self.prefix + self.enc(f"<|system|>\n{system}<|user|>\n") + user_ids + self.enc(head))

    def interleave(self, text_ids, audio_toks):
        """13 text tokens, 26 audio tokens, ... then whatever remains of either (web_demo.py order)."""
        out, ti, ai = [], 0, 0
        audio_ids = [self.audio_offset + t for t in audio_toks]
        while ti < len(text_ids) or ai < len(audio_ids):
            if ti < len(text_ids):
                out += text_ids[ti:ti + TEXT_CHUNK]; ti += TEXT_CHUNK
                out += audio_ids[ai:ai + AUDIO_CHUNK]; ai += AUDIO_CHUNK
            else:
                out += audio_ids[ai:]; ai = len(audio_ids)
        return out

    def sample(self, prompt_ids, target_ids):
        ids = prompt_ids + target_ids + [self.user_id]           # <|user|> = end of turn (stop token)
        labels = [-100] * len(prompt_ids) + target_ids + [self.user_id]
        return ids, labels
