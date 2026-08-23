"""Open-weight VLM answering arm.

Deliberately *not* a :class:`~quantiphy.vision.VisionBackend`. That Protocol returns a
``PixelMeasurement`` -- pixels, for the geometric solver to scale -- and this returns an answer in
the question's own unit. Forcing one interface over both would make the type lie about what it
produces; the two are combined by :mod:`quantiphy.fusion` rather than by substitution.

Why this arm exists: the zero-vision constant is capped around 0.38 and the geometric solver's
realistic ceiling is ~0.42, while Qwen3-VL-32B published **46.0** on this exact test split. The
hypothesis for *beating* that number rather than reproducing it lives in :mod:`quantiphy.prompting`
-- every row carries a real scene measurement that the benchmark paper shows VLMs ignoring, and we
hand it over explicitly.

Model loading goes through the ``Auto*`` classes rather than a named architecture, so a Qwen3-VL, an
InternVL or a Llava-style checkpoint all load by the same path and swapping models is a config
change. torch and transformers are imported lazily inside the methods, which keeps this module -- and
the whole test suite -- importable on a machine with no GPU and no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Frames handed to the model per question. Vision prefill dominates the cost (the reply is one
#: number), so this is the main cost/accuracy dial. 12 at ~250 visual tokens each is ~3k tokens/row.
DEFAULT_FRAMES = 12

#: Half-width, in seconds, of the window frames are drawn from when the question names an instant.
#: Same reasoning as ``grounding.FIT_WINDOW_S``: a question about t=1.4 s is not answered better by
#: frames from t=15 s, and spending the budget locally is more accurate at no extra cost.
INSTANT_WINDOW_S = 1.0


@dataclass
class VlmAnswer:
    """One model reply, kept raw.

    ``raw_text`` is the payload; everything else is provenance. Parsing is *not* done here on
    purpose -- the run caches raw text so prompt-parsing and fusion changes can be re-measured
    offline for free, which is the discipline that has saved this project the most money.
    """

    row_index: int
    video_id: str
    raw_text: str
    prompt: str = ""
    frame_times: tuple[float, ...] = field(default_factory=tuple)
    model: str = ""
    note: str = ""


def choose_frame_times(times: np.ndarray, at_time: float | None, count: int) -> np.ndarray:
    """Indices of the frames to show the model.

    Uniform across the clip when no instant is named: the question is about the whole scene, and a
    size question is answered from whichever frame shows the object best. Concentrated within
    ``INSTANT_WINDOW_S`` of the instant when one *is* named, because that is where the answer lives.
    Falls back to uniform when the window cannot fill the budget, rather than showing three frames
    where twelve were affordable.
    """
    if times.size == 0:
        return np.empty(0, dtype=int)
    count = min(count, times.size)
    if at_time is not None:
        near = np.flatnonzero(np.abs(times - at_time) <= INSTANT_WINDOW_S)
        if near.size >= count:
            return near[np.linspace(0, near.size - 1, count).astype(int)]
    return np.linspace(0, times.size - 1, count).astype(int)


class VlmBackend:
    """A vision-language model answering one question at a time."""

    def __init__(self, model_id: str, frames: int = DEFAULT_FRAMES,
                 max_new_tokens: int = 128, load_in_4bit: bool = False) -> None:
        self.model_id = model_id
        self.frames = frames
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None

    # ---------------------------------------------------------------- model

    @staticmethod
    def _preferred_dtype():
        """bfloat16 where the GPU actually supports it, float16 otherwise.

        Not a detail. The free tiers this arm is meant to run on are Turing and Pascal -- Kaggle
        offers T4 x2 and P100, and neither has native bf16 -- while the paid HF Jobs flavours are
        Ampere or newer and do. Hardcoding bf16 either crawls through emulation or fails outright on
        exactly the hardware we chose in order to spend nothing.
        """
        import torch
        if not torch.cuda.is_available():
            return torch.float32
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = self._preferred_dtype()
        kwargs: dict = {"dtype": dtype, "device_map": "auto"}
        if self.load_in_4bit:
            # 4-bit exists to fit a 32B on hardware already paid for -- Kaggle's 2x16 GB, or a 40 GB
            # A100. Whether it costs accuracy is a measurement, not an assumption: run it against the
            # 159 validation rows alongside full precision before committing a test pass to it.
            from transformers import BitsAndBytesConfig
            kwargs.pop("dtype")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        self._model.eval()

    @property
    def device(self) -> str:
        self._load()
        return str(next(self._model.parameters()).device)

    # ---------------------------------------------------------------- frames

    def read_frames(self, video_path: str, at_time: float | None):
        """Sampled PIL frames and their true timestamps.

        Timestamps come from the real frame index over the container fps, never from the sampled
        position -- the same correctness point that makes the kinematics fit work.
        """
        import cv2
        from PIL import Image

        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            capture.release()
            return [], np.empty(0)

        wanted = set(choose_frame_times(np.arange(total) / fps, at_time, self.frames).tolist())
        frames, times, index = [], [], 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                times.append(index / fps)
            index += 1
        capture.release()
        return frames, np.asarray(times, dtype=float)

    # ---------------------------------------------------------------- answer

    def answer(self, row_index: int, video_id: str, video_path: str, system: str, prompt: str,
               at_time: float | None) -> VlmAnswer:
        """Ask one question. Returns the reply unparsed."""
        import torch

        frames, times = self.read_frames(video_path, at_time)
        if not frames:
            return VlmAnswer(row_index, video_id, "", prompt, (), self.model_id,
                             note="no frames could be read")

        self._load()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image"} for _ in frames]
                                        + [{"type": "text", "text": prompt}]},
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = self._processor(text=[text], images=frames, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                             do_sample=False)
        # Slice off the prompt before decoding. Decoding the whole sequence would feed our own
        # instructions to the parser, and "ANSWER: <number>" appears in the prompt itself.
        reply = self._processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return VlmAnswer(row_index, video_id, reply.strip(), prompt,
                         tuple(times.tolist()), self.model_id)
