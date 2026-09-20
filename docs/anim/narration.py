"""Narration for the combined overview video.

One entry per scene, in order. `text` is read verbatim by the TTS voice;
keep it close to the scene's own length — the renderer freezes the last frame
to cover whatever is left over.

    docs/anim/narrate.sh        # generates docs/media/narration/*.m4a
"""

NARRATION = [
    ("S1Pipeline", """GLM-4-Voice is not one model. It is four.
A frozen tokenizer turns your speech into discrete audio tokens.
The nine billion parameter language model reads those tokens, and answers in
text and audio tokens of its own. A flow decoder turns the audio tokens into a
mel spectrogram, and a frozen vocoder turns that into sound.
Only the middle two boxes were trained."""),

    ("S2Tokenizer", """Everything downstream is integers.
Eighty milliseconds of audio goes in, and one integer comes out, twelve and a
half times a second. Each frame is matched to its nearest entry in a codebook of
sixteen thousand three hundred and eighty four codes, and the index of that entry
is the token. A ten second clip becomes a hundred and twenty five integers.
They carry what is said, and almost nothing about who said it."""),

    ("S4LoRA", """The nine B learns Somali through a one point seven percent patch.
Its own weights never move. Two thin matrices beside every attention and
M.L.P. projection do all the learning: rank sixty four, a hundred and sixty nine
million weights, six hundred and seventy eight megabytes in total.
A full fine tune would need optimizer state for all nine billion weights,
far more than one ninety six gigabyte card holds. Before inference,
the adapter is folded back into the base model."""),

    ("S3Interleave", """Every training sample uses the exact chat format the demo
uses at inference. The prompt is masked out; only the answer is learned.
And inside that answer, the model writes thirteen text tokens, then the twenty
six audio tokens that speak them, about two seconds of sound, then the next
thirteen. That rhythm is the one it was pre-trained with, so we teach it Somali
without re-teaching it how to stream."""),

    ("S5Voice", """Content and voice are trained apart.
Track A: a hundred and sixteen hours from five Somali channels, many speakers,
teaches the language model what is said. Track B: sixty one hours of one
podcaster teaches the flow decoder how it sounds.
Separate data, separate weights. They meet for the first time at inference,
where the decoder gives Omar's voice to whatever the model decided to say."""),
]
