"""Manim scenes for docs/index.html.

Five animations explaining the four models of GLM-4-Voice and how audio tokens
move through them. Rendered twice, once per colour theme:

    SO_THEME=light manim -qh --format=mp4 scenes.py S1Pipeline
    SO_THEME=dark  manim -qh --format=mp4 scenes.py S1Pipeline

See docs/anim/render.sh, which renders every scene in both themes into
docs/media/.
"""

import os
import random

from manim import *

THEME = os.environ.get("SO_THEME", "dark")

# Pacing: seconds of stillness inserted after every animation, so a scene can be
# stretched to fit its narration without slowing the motion itself down.
# docs/anim/render.sh measures each scene once and then picks this value.
GAP = float(os.environ.get("SO_GAP", "0"))

if THEME == "light":
    BG, FG, MUTED, BORDER, SOFT = "#ffffff", "#1f2328", "#656d76", "#d8dee4", "#f6f8fa"
    FROZEN, LORA_C, FULL_C = "#6e7781", "#8250df", "#1a7f37"
    TEXTTOK, AUDIOTOK, JUDGE = "#0969da", "#d4760a", "#bf5b04"
    BAD, GOOD = "#cf222e", "#1a7f37"
    FILL = 0.06
else:
    BG, FG, MUTED, BORDER, SOFT = "#0d1117", "#e6edf3", "#9198a1", "#30363d", "#151b23"
    FROZEN, LORA_C, FULL_C = "#9198a1", "#b391f5", "#3fb950"
    TEXTTOK, AUDIOTOK, JUDGE = "#4493f8", "#f0a04b", "#f0883e"
    BAD, GOOD = "#ff7b72", "#3fb950"
    FILL = 0.14

FONT = "Helvetica Neue"
MONO = "Menlo"


class Base(Scene):
    def setup(self):
        self.camera.background_color = BG
        self._plays = 0
        self._pausing = False

    def construct(self):
        self.body()
        # render.sh reads this to work out the gap for the final pass
        print("[pace] %s plays=%d" % (type(self).__name__, self._plays))

    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        if self._pausing:      # Scene.wait() routes back through play()
            return
        self._plays += 1
        if GAP:
            self._pausing = True
            try:
                self.wait(GAP)
            finally:
                self._pausing = False

    # ---------- small builders ----------

    def t(self, s, size=24, color=None, weight=NORMAL, mono=False):
        return Text(
            s,
            font=MONO if mono else FONT,
            font_size=size,
            color=color or FG,
            weight=weight,
        )

    def title(self, s, sub=None):
        g = VGroup(self.t(s, 34, FG, BOLD))
        if sub:
            g.add(self.t(sub, 20, MUTED))
            g.arrange(DOWN, buff=0.18)
        g.to_edge(UP, buff=0.45)
        return g

    def box(self, head, body, accent, tag=None, w=3.0, h=2.0):
        """A labelled model box: tag pill, heading, one-line description."""
        rect = RoundedRectangle(
            corner_radius=0.12, width=w, height=h,
            stroke_color=accent, stroke_width=2,
            fill_color=accent, fill_opacity=FILL,
        )
        parts = VGroup()
        if tag:
            parts.add(self.t(tag.upper(), 13, accent, BOLD))
        parts.add(self.t(head, 21, FG, BOLD))
        for line in body:
            parts.add(self.t(line, 15, MUTED))
        parts.arrange(DOWN, buff=0.13).move_to(rect)
        g = VGroup(rect, parts)
        g.rect, g.accent = rect, accent
        return g

    def wave(self, n=60, width=2.2, height=0.9, color=None, seed=3):
        """A little waveform drawn as vertical bars."""
        rng = random.Random(seed)
        bars = VGroup()
        for i in range(n):
            env = np.sin(np.pi * (i + 0.5) / n) ** 0.7
            a = max(0.06, env * (0.35 + 0.65 * rng.random()))
            bars.add(Line(
                [0, -a * height / 2, 0], [0, a * height / 2, 0],
                stroke_width=2.4, color=color or FG,
            ))
        bars.arrange(RIGHT, buff=width / n * 0.35)
        bars.stretch_to_fit_width(width)
        return bars

    def tok(self, label, color, w=0.92, h=0.44, size=13):
        r = RoundedRectangle(
            corner_radius=0.07, width=w, height=h,
            stroke_color=color, stroke_width=1.6,
            fill_color=color, fill_opacity=FILL + 0.1,
        )
        return VGroup(r, self.t(label, size, color, mono=True).move_to(r))

    def arrow(self, a, b, color=None, label=None, size=13):
        ar = Arrow(
            a.get_right(), b.get_left(), buff=0.16,
            stroke_width=3, color=color or MUTED,
            max_tip_length_to_length_ratio=0.22,
        )
        if label is None:
            return ar
        lb = self.t(label, size, MUTED).next_to(ar, UP, buff=0.1)
        return VGroup(ar, lb)

    def mel(self, cols=26, rows=10, w=2.4, h=1.1, seed=7, color=None):
        """A mel spectrogram as a grid of cells with varying opacity."""
        rng = random.Random(seed)
        c = color or FULL_C
        cells = VGroup()
        for i in range(cols):
            col = VGroup()
            base = rng.random()
            for j in range(rows):
                e = np.exp(-((j - rows * 0.3) ** 2) / (2 * (rows * 0.35) ** 2))
                v = min(1.0, 0.12 + e * (0.35 + 0.9 * base) * (0.6 + 0.8 * rng.random()))
                col.add(Square(
                    side_length=0.2, stroke_width=0,
                    fill_color=c, fill_opacity=v,
                ))
            col.arrange(UP, buff=0.012)
            cells.add(col)
        cells.arrange(RIGHT, buff=0.012)
        cells.stretch_to_fit_width(w)
        cells.stretch_to_fit_height(h)
        return cells

    def caption(self, s, size=18, color=None):
        return self.t(s, size, color or MUTED).to_edge(DOWN, buff=0.5)


# ------------------------------------------------------------------ 1
class S1Pipeline(Base):
    """The four models, and one utterance travelling through all of them."""

    def body(self):
        ttl = self.title("GLM-4-Voice is not one model. It is four.",
                         "Speech in, speech out — each box speaks a different language")
        self.play(FadeIn(ttl, shift=DOWN * 0.2))

        tokz = self.box("Speech tokenizer", ["Whisper-VQ, 16,384 codes",
                                             "audio → 12.5 tok/s"],
                        FROZEN, "frozen", w=2.9, h=2.0)
        llm = self.box("GLM-4-Voice 9B", ["the brain",
                                          "tokens → text + audio"],
                       LORA_C, "LoRA", w=2.9, h=2.0)
        flow = self.box("Flow decoder", ["the voice lives here",
                                         "audio tok → 80-bin mel"],
                        FULL_C, "full fine-tune", w=2.9, h=2.0)
        hift = self.box("HiFT vocoder", ["works for any speaker",
                                         "mel → 22.05 kHz wav"],
                        FROZEN, "frozen", w=2.9, h=2.0)

        row = VGroup(tokz, llm, flow, hift).arrange(RIGHT, buff=0.62)

        wav_in = self.wave(40, 1.1, 0.8, TEXTTOK, seed=11)
        wav_out = self.wave(40, 1.1, 0.8, FULL_C, seed=5)
        lab_in = self.t("you speak", 15, MUTED)
        lab_out = self.t("it replies", 15, MUTED)
        in_g = VGroup(wav_in, lab_in).arrange(DOWN, buff=0.22)
        out_g = VGroup(wav_out, lab_out).arrange(DOWN, buff=0.22)

        whole = VGroup(in_g, row, out_g).arrange(RIGHT, buff=0.42)
        whole.set(width=13.4).move_to(ORIGIN).shift(DOWN * 0.55)

        arrows = [self.arrow(a, b) for a, b in
                  ((wav_in, tokz), (tokz, llm), (llm, flow), (flow, hift), (hift, wav_out))]

        self.play(FadeIn(in_g), run_time=0.6)
        for b, a in zip((tokz, llm, flow, hift), arrows):
            self.play(GrowArrow(a), FadeIn(b, shift=RIGHT * 0.2), run_time=0.55)
        self.play(GrowArrow(arrows[4]), FadeIn(out_g), run_time=0.6)
        self.wait(0.4)

        note = self.caption("Only the middle two boxes were trained. "
                            "The tokenizer and the vocoder are used exactly as they shipped.")
        self.play(FadeIn(note), run_time=0.5)
        self.wait(1.2)

        # one packet of audio tokens travels the whole chain
        packet = VGroup(*[
            self.tok(s, AUDIOTOK, w=0.8, h=0.38, size=12)
            for s in ("10815", "10057", "7988")
        ]).arrange(RIGHT, buff=0.07)
        packet.move_to(tokz.get_center()).set_opacity(0)

        def mid(a, b):
            # float the packet above the row so it never lands on a box
            return np.array([(a.get_right()[0] + b.get_left()[0]) / 2,
                             row.get_top()[1] + 0.62, 0.0])

        self.play(packet.animate.set_opacity(1).move_to(mid(tokz, llm)), run_time=0.7)
        self.play(Indicate(llm.rect, color=LORA_C, scale_factor=1.04), run_time=0.6)

        reply = VGroup(
            self.tok("waa", TEXTTOK, w=0.62, h=0.38, size=12),
            self.tok("4471", AUDIOTOK, w=0.72, h=0.38, size=12),
            self.tok("903", AUDIOTOK, w=0.66, h=0.38, size=12),
        ).arrange(RIGHT, buff=0.07).move_to(mid(llm, flow))
        self.play(Transform(packet, reply), run_time=0.8)

        self.play(Indicate(flow.rect, color=FULL_C, scale_factor=1.04), run_time=0.5)
        melg = self.mel(22, 9, 1.1, 0.66).move_to(mid(flow, hift))
        self.play(Transform(packet, melg), run_time=0.8)

        self.play(Indicate(hift.rect, color=FROZEN, scale_factor=1.04), run_time=0.5)
        self.play(Transform(packet, wav_out.copy().set_color(FULL_C)), run_time=0.9)
        self.play(FadeOut(packet), Flash(wav_out, color=FULL_C, line_length=0.18),
                  run_time=0.6)
        self.wait(1.4)


# ------------------------------------------------------------------ 2
class S2Tokenizer(Base):
    """Continuous audio becomes 12.5 discrete integers per second."""

    def body(self):
        ttl = self.title("Step 1: sound becomes integers",
                         "The frozen Whisper-VQ tokenizer, 12.5 tokens per second")
        self.play(FadeIn(ttl, shift=DOWN * 0.2))

        w = self.wave(120, 7.4, 1.5, TEXTTOK, seed=21).shift(UP * 1.35 + LEFT * 2.2)
        self.play(Create(w, lag_ratio=0.01), run_time=1.2)

        secs = self.t("one second of speech = 12.5 tokens = 80 ms per token",
                      16, MUTED).next_to(w, DOWN, buff=0.3)
        self.play(FadeIn(secs), run_time=0.5)

        # slice the waveform into frames
        n_frames = 10
        edges = VGroup()
        fw = w.width / n_frames
        for i in range(n_frames + 1):
            x = w.get_left()[0] + i * fw
            edges.add(DashedLine(
                [x, w.get_top()[1] + 0.12, 0], [x, w.get_bottom()[1] - 0.12, 0],
                stroke_width=1.4, color=BORDER, dash_length=0.06,
            ))
        self.play(Create(edges, lag_ratio=0.12), run_time=1.0)

        # the codebook
        cb_rows, cb_cols = 8, 20
        cb = VGroup(*[
            Square(side_length=0.18, stroke_width=0.6, stroke_color=BORDER,
                   fill_color=AUDIOTOK, fill_opacity=0.05)
            for _ in range(cb_rows * cb_cols)
        ]).arrange_in_grid(rows=cb_rows, cols=cb_cols, buff=0.035)
        cb.scale(0.9).shift(DOWN * 1.3 + LEFT * 2.2)
        cb_lab = VGroup(
            self.t("codebook", 19, FG, BOLD),
            self.t("16,384 entries", 15, MUTED),
            self.t("already in the 9B's vocabulary", 14, MUTED),
        ).arrange(DOWN, buff=0.1).next_to(cb, DOWN, buff=0.25)

        self.play(FadeIn(cb, lag_ratio=0.004), FadeIn(cb_lab), run_time=1.1)

        out_lab = self.t("audio tokens", 19, AUDIOTOK, BOLD)
        out = VGroup()
        ids = ["10815", "10057", "7988", "8342", "2946"]
        for s in ids:
            out.add(self.tok(f"<|audio_{s}|>", AUDIOTOK, w=2.5, h=0.44, size=14))
        out.arrange(DOWN, buff=0.14)
        col = VGroup(out_lab, out).arrange(DOWN, buff=0.22)
        col.shift(RIGHT * 4.55 + DOWN * 0.55)
        self.play(FadeIn(out_lab), run_time=0.4)

        rng = random.Random(4)
        for k in range(5):
            x0 = w.get_left()[0] + k * fw
            hl = Rectangle(
                width=fw, height=w.height + 0.24,
                stroke_width=0, fill_color=AUDIOTOK, fill_opacity=0.22,
            ).move_to([x0 + fw / 2, w.get_center()[1], 0])
            cell = cb[rng.randrange(cb_rows * cb_cols)]
            line = Line(hl.get_bottom(), cell.get_top(), stroke_width=1.6,
                        color=AUDIOTOK, stroke_opacity=0.55)
            self.play(FadeIn(hl), run_time=0.22)
            self.play(Create(line), cell.animate.set_fill(AUDIOTOK, opacity=0.95),
                      run_time=0.3)
            self.play(FadeIn(out[k], shift=RIGHT * 0.25), run_time=0.28)
            self.play(FadeOut(hl), FadeOut(line),
                      cell.animate.set_fill(AUDIOTOK, opacity=0.45), run_time=0.22)

        note = self.caption("A 10-second clip becomes 125 integers. "
                            "They carry what is said — and almost nothing about who said it.")
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.6)


# ------------------------------------------------------------------ 3
class S3Interleave(Base):
    """The 13:26 streaming rhythm and the loss mask."""

    def body(self):
        ttl = self.title("How the model answers: 13 text, then 26 audio",
                         "The exact rhythm GLM-4-Voice was pre-trained with")
        self.play(FadeIn(ttl, shift=DOWN * 0.2))

        # the chat template, prompt masked / target learned
        prompt = self.t("[gMASK]<sop><|system|>\\n{system}<|user|>\\n{audio}<|assistant|>streaming\\n",
                        18, MUTED, mono=True)
        target = self.t("{answer}<|user|>", 18, FULL_C, mono=True)
        line = VGroup(prompt, target).arrange(RIGHT, buff=0.12)
        line.set(width=11.6).shift(UP * 1.95)
        pbox = SurroundingRectangle(prompt, buff=0.1, stroke_width=1.2,
                                    color=BORDER, fill_color=SOFT, fill_opacity=0.6)
        tbox = SurroundingRectangle(target, buff=0.1, stroke_width=1.4,
                                    color=FULL_C, fill_color=FULL_C, fill_opacity=FILL)
        self.play(FadeIn(pbox), FadeIn(prompt), run_time=0.5)
        self.play(FadeIn(tbox), FadeIn(target), run_time=0.5)

        k1 = self.t("loss masked  (−100)", 14, MUTED).next_to(pbox, DOWN, buff=0.18)
        k2 = self.t("learned", 14, FULL_C).next_to(tbox, DOWN, buff=0.18)
        self.play(FadeIn(k1), FadeIn(k2), run_time=0.4)
        self.wait(0.5)

        stop = self.t("that final <|user|> is how the model says \"my turn is over\" — "
                      "without it, it never stops talking", 16, MUTED)
        stop.next_to(line, DOWN, buff=0.75)
        self.play(FadeIn(stop), run_time=0.5)
        self.wait(0.8)

        # the interleaved strip
        strip = VGroup()
        for _ in range(3):
            for _ in range(13):
                strip.add(Rectangle(width=0.13, height=0.46, stroke_width=0,
                                    fill_color=TEXTTOK, fill_opacity=0.9))
            for _ in range(26):
                strip.add(Rectangle(width=0.13, height=0.46, stroke_width=0,
                                    fill_color=AUDIOTOK, fill_opacity=0.9))
        strip.arrange(RIGHT, buff=0.028).shift(DOWN * 0.75)
        strip.set_width(12.6)

        self.play(LaggedStart(*[FadeIn(b, scale=0.6) for b in strip],
                              lag_ratio=0.014), run_time=2.6)

        b1 = Brace(VGroup(*strip[0:13]), DOWN, buff=0.12, color=TEXTTOK)
        l1 = self.t("13 text tokens", 15, TEXTTOK).next_to(b1, DOWN, buff=0.1).shift(LEFT * 0.45)
        b2 = Brace(VGroup(*strip[13:39]), DOWN, buff=0.12, color=AUDIOTOK)
        l2 = self.t("26 audio tokens  ≈ 2 seconds of speech", 15, AUDIOTOK
                    ).next_to(b2, DOWN, buff=0.1).shift(RIGHT * 0.7)
        self.play(GrowFromCenter(b1), FadeIn(l1), run_time=0.5)
        self.play(GrowFromCenter(b2), FadeIn(l2), run_time=0.5)
        self.wait(0.6)

        words = self.t("\"Waa\"    \"yahay, waxaan\"    \"filayaa in…\"", 18, TEXTTOK
                       ).next_to(strip, UP, buff=0.45)
        self.play(FadeIn(words), run_time=0.5)

        note = self.caption("It writes a little text, speaks it, writes more. "
                            "Keeping the ratio means we teach Somali, not streaming.")
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.6)


# ------------------------------------------------------------------ 4
class S4LoRA(Base):
    """Why the 9B is adapted, not retrained."""

    def body(self):
        ttl = self.title("The 9B learns Somali through a 1.7% patch",
                         "LoRA: rank 64, alpha 128, on every attention and MLP projection")
        self.play(FadeIn(ttl, shift=DOWN * 0.2))

        W = Square(side_length=2.9, stroke_color=FROZEN, stroke_width=2,
                   fill_color=FROZEN, fill_opacity=FILL)
        wl = VGroup(
            self.t("W", 30, FROZEN, BOLD),
            self.t("frozen", 15, MUTED),
            self.t("9B weights", 14, MUTED),
        ).arrange(DOWN, buff=0.1).move_to(W)
        Wg = VGroup(W, wl)

        plus = self.t("+", 34, MUTED)

        B = Rectangle(width=0.55, height=2.9, stroke_color=LORA_C, stroke_width=2,
                      fill_color=LORA_C, fill_opacity=FILL + 0.08)
        A = Rectangle(width=2.9, height=0.55, stroke_color=LORA_C, stroke_width=2,
                      fill_color=LORA_C, fill_opacity=FILL + 0.08)
        BA = VGroup(B, A).arrange(RIGHT, buff=0.18)
        bl = self.t("B", 20, LORA_C, BOLD).move_to(B)
        al = self.t("A", 20, LORA_C, BOLD).move_to(A)
        BAg = VGroup(BA, bl, al)
        rlab = self.t("r = 64", 16, LORA_C).next_to(BA, DOWN, buff=0.22)

        eq = VGroup(Wg, plus, VGroup(BAg, rlab)).arrange(RIGHT, buff=0.6)
        eq.shift(LEFT * 2.6 + DOWN * 0.5)

        self.play(FadeIn(Wg, scale=0.95), run_time=0.7)
        self.play(FadeIn(plus), FadeIn(BAg, shift=LEFT * 0.2), FadeIn(rlab), run_time=0.7)

        def stat(big, small, color):
            return VGroup(self.t(big, 26, color, BOLD),
                          self.t(small, 15, MUTED)).arrange(DOWN, buff=0.1)

        stats = VGroup(
            stat("9,000,000,000", "weights frozen", FROZEN),
            stat("169,000,000", "weights trained  ·  1.7%", LORA_C),
            stat("678 MB", "the whole Somali adapter", LORA_C),
        ).arrange(DOWN, buff=0.5)
        stats.shift(RIGHT * 4.3 + DOWN * 0.5)
        self.play(FadeIn(stats, shift=UP * 0.2), run_time=0.9)
        self.wait(1.0)

        why = self.caption("A full fine-tune of 9B needs optimizer state for every weight — "
                           "far more than one 96 GB card holds.")
        self.play(FadeIn(why), run_time=0.6)
        self.wait(1.4)
        self.play(FadeOut(why), run_time=0.3)

        # merge
        merged = Square(side_length=2.9, stroke_color=LORA_C, stroke_width=2,
                        fill_color=LORA_C, fill_opacity=FILL)
        ml = VGroup(
            self.t("W + BA", 26, LORA_C, BOLD),
            self.t("merged for inference", 15, MUTED),
        ).arrange(DOWN, buff=0.12).move_to(merged)
        mg = VGroup(merged, ml).move_to(Wg)

        self.play(
            BAg.animate.move_to(Wg).set_opacity(0),
            FadeOut(rlab), FadeOut(plus),
            Transform(Wg, mg),
            run_time=1.2,
        )
        note = self.caption("--scale 0.3 folds in a third of the delta: "
                            "a weaker Somali accent, more of the base model's manners.")
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.6)


# ------------------------------------------------------------------ 5
class S5Voice(Base):
    """Content and voice are learned separately, and meet only at inference."""

    def body(self):
        ttl = self.title("Content and voice are trained apart",
                         "They meet for the first time at inference")
        self.play(FadeIn(ttl, shift=DOWN * 0.2))

        # Track A
        many = VGroup(*[self.wave(26, 0.9, 0.42, TEXTTOK, seed=s) for s in (1, 2, 3)])
        many.arrange(DOWN, buff=0.18)
        ta_lab = VGroup(
            self.t("Track A", 17, TEXTTOK, BOLD),
            self.t("116 h, 5 channels,", 14, MUTED),
            self.t("many speakers", 14, MUTED),
        ).arrange(DOWN, buff=0.06).next_to(many, DOWN, buff=0.22)
        A_side = VGroup(many, ta_lab).shift(LEFT * 5.3 + UP * 1.55)

        llm = self.box("9B + LoRA", ["learns WHAT is said"], LORA_C, "LoRA", w=3.2, h=1.45)
        llm.move_to(LEFT * 0.9 + UP * 1.9)

        # Track B
        omar = self.wave(40, 1.6, 0.6, FULL_C, seed=9)
        tb_lab = VGroup(
            self.t("Track B", 17, FULL_C, BOLD),
            self.t("61 h of Omar alone,", 14, MUTED),
            self.t("8,163 verified clips", 14, MUTED),
        ).arrange(DOWN, buff=0.06).next_to(omar, DOWN, buff=0.22)
        B_side = VGroup(omar, tb_lab).shift(LEFT * 5.3 + DOWN * 1.75)

        flow = self.box("Flow decoder", ["learns HOW it sounds"], FULL_C,
                        "full fine-tune", w=3.2, h=1.45)
        flow.move_to(LEFT * 0.9 + DOWN * 1.5)

        self.play(FadeIn(A_side), FadeIn(B_side), run_time=0.8)
        self.play(GrowArrow(self.arrow(many, llm)), FadeIn(llm), run_time=0.6)
        self.play(GrowArrow(self.arrow(omar, flow)), FadeIn(flow), run_time=0.6)

        div = DashedLine(RIGHT * 1.15 + UP * 3.0, RIGHT * 1.15 + DOWN * 2.6,
                         stroke_width=1.4, color=BORDER, dash_length=0.1)
        dlab = self.t("separate data · separate weights", 13, MUTED)
        dlab.rotate(PI / 2).move_to(div.get_center() + DOWN * 0.9 + RIGHT * 0.18)
        self.play(Create(div), FadeIn(dlab), run_time=0.6)
        self.wait(0.8)

        # inference: they meet
        meet = self.t("at inference", 16, MUTED)
        meet.shift(RIGHT * 3.7 + UP * 2.6)
        toks = VGroup(*[self.tok(s, AUDIOTOK, w=0.8, h=0.38, size=11)
                        for s in ("4471", "903", "12088")]).arrange(RIGHT, buff=0.08)
        toks.shift(RIGHT * 3.7 + UP * 1.95)
        feed = Arrow(llm.get_right(), toks.get_left(), buff=0.18, stroke_width=3,
                     color=AUDIOTOK, max_tip_length_to_length_ratio=0.16)
        self.play(FadeIn(meet), GrowArrow(feed), FadeIn(toks, shift=RIGHT * 0.2),
                  run_time=0.8)

        down = Arrow(toks.get_bottom(), toks.get_bottom() + DOWN * 0.8, buff=0.06,
                     stroke_width=3, color=AUDIOTOK,
                     max_tip_length_to_length_ratio=0.3)
        melg = self.mel(30, 11, 3.0, 1.05).next_to(down, DOWN, buff=0.1)
        mlab = self.t("mel spectrogram — Omar's timbre", 14, FULL_C
                      ).next_to(melg, DOWN, buff=0.14)
        voice = DashedLine(flow.get_right() + RIGHT * 0.1, down.get_center() + LEFT * 0.15,
                           stroke_width=1.6, color=FULL_C, dash_length=0.09)
        vlab = self.t("supplies the voice for whatever the LLM says", 13, FULL_C)
        vlab.next_to(flow, DOWN, buff=0.22)

        self.play(GrowArrow(down), run_time=0.4)
        self.play(Create(voice), FadeIn(vlab), run_time=0.7)
        self.play(FadeIn(melg, lag_ratio=0.01), FadeIn(mlab), run_time=0.9)

        down2 = Arrow(mlab.get_bottom(), mlab.get_bottom() + DOWN * 0.5, buff=0.06,
                      stroke_width=3, color=FULL_C,
                      max_tip_length_to_length_ratio=0.34)
        out = self.wave(56, 3.0, 0.7, FULL_C, seed=17).next_to(down2, DOWN, buff=0.12)
        self.play(GrowArrow(down2), run_time=0.35)
        self.play(Create(out, lag_ratio=0.01), run_time=0.8)
        self.play(Flash(out, color=FULL_C, line_length=0.16), run_time=0.5)

        note = self.t("Somali learned from everybody. One voice learned from one person.",
                      18, MUTED).to_edge(DOWN, buff=0.3).shift(LEFT * 2.6)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.6)
