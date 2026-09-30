"""Detection code shared by the live demo (demo/app.py) and scripts/prebake.py."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# offline / quiet settings. These have to be set before ultralytics is imported, otherwise it tries
# to reach the network (update checks, telemetry, font download, Hugging Face calls).
os.environ.setdefault("YOLO_OFFLINE", "1")  # ultralytics: treat as offline (no GitHub / telemetry calls)
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "weights"
IMAGES = REPO / "demo" / "images"

import ultralytics.nn.text_model as _tm
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils import SETTINGS

_tm.WEIGHTS_DIR = WEIGHTS  # CLIP ViT-B/32 is looked up in <repo>/weights/clip, independent of the CWD
try:
    SETTINGS["sync"] = False  # in-memory only: disables anonymous usage events for this process
except Exception:  # settings object may be read-only in some versions
    pass

SIZES = ("S", "M", "L")
_FORCE_CPU = os.environ.get("DEMO_DEVICE", "").lower() == "cpu"  # set DEMO_DEVICE=cpu to bypass the GPU
DEVICE = 0 if (torch.cuda.is_available() and not _FORCE_CPU) else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == 0 else "CPU"


def _gpu_responsive(timeout_s: float = 60.0) -> bool:
    # Runs a small cuBLAS matmul in a separate process with a timeout.
    # On my Windows laptop CUDA sometimes stalled forever (every new process hung in its first cuBLAS call
    # while another hung CUDA process was still around). A stalled call can't be cancelled from inside the
    # process, so the probe runs in a child process that can be killed, and the demo falls back to the CPU
    # instead of freezing. Skip with DEMO_SKIP_GPU_CHECK=1.
    import subprocess
    import sys

    code = ("import torch; a = torch.randn(1024, 1024, device='cuda'); "
            "print(float((a @ a).sum().item()) == float((a @ a).sum().item()))")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout_s,
                           stdin=subprocess.DEVNULL)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return True  # could not run the probe for an unrelated reason -> do not block the GPU path


if DEVICE == 0 and os.environ.get("DEMO_SKIP_GPU_CHECK", "") != "1":
    if not _gpu_responsive():
        print("WARNING: the GPU did not finish a small CUDA test within 60 s (another process may be stuck on it).\n"
              "         Falling back to CPU so the demo still works. Close other GPU apps / reboot to get CUDA back.")
        DEVICE, DEVICE_NAME = "cpu", "CPU (GPU unresponsive, fell back)"

GPU_CALL_TIMEOUT_S = 30.0  # the app reports an error instead of spinning forever if one inference stalls

DEFAULT_VOCAB = "person, guitar, microphone, helmet, sunglasses, flag, lamp, traffic cone, statue, backpack"
MAX_CLASSES = 200  # hard cap so a runaway paste cannot blow up GPU memory
DEFAULT_CONF = 0.10  # YOLO-World scores run lower than closed-set scores; 0.05-0.2 works well
DEFAULT_CONF_CLOSED = 0.25  # Ultralytics' default for YOLOv8

# Sample images (demo/images, see ATTRIBUTION.md) + a vocabulary that suits each one.
SAMPLES = [
    {"file": "guitar_stage.jpg", "vocab": "person, guitar, microphone, necktie", "failure_case": True,
     "note": ("Miss: YOLO-World-S does not report the necktie at the 0.10 threshold (its best 'necktie' "
              "score is only ~0.06), while the closed-set YOLOv8s finds it as COCO class 'tie' (~0.6). "
              "Open vocabulary does not mean better on every class the closed-set model was trained on.")},
    {"file": "traffic_cones.jpg", "vocab": "traffic cone, truck, car, street lamp, crane", "conf": 0.2},
    {"file": "bike_helmet.jpg", "vocab": "person, bicycle, helmet, sneakers"},
    {"file": "podcast_bear.jpg", "vocab": "teddy bear, glasses, microphone, keyboard, mouse, ipod"},
    {"file": "flags_street.jpg", "vocab": "flag, street sign, traffic light, skyscraper"},
    {"file": "bedroom_lamp.jpg", "vocab": "lamp, bed, book, pillow, nightstand"},
    {"file": "living_room.jpg", "vocab": "piano, picture frame, sofa, coffee table, lamp", "conf": 0.3,
     "note": ("Threshold raised to 0.3 to hide weak duplicates. Adding 'fireplace' to this vocabulary makes "
              "YOLO-World-S label the piano as 'fireplace' (0.51) - a second, milder failure mode.")},
    {"file": "statue_bench.jpg", "vocab": "statue, person, bench, handbag", "failure_case": True,
     "note": ("Deliberate failure case: the bronze statues are labelled 'person' (by both models) and "
              "'statue' is never output while 'person' is also in the vocabulary. Region-text matching "
              "picks the more frequent training concept; try removing 'person' from the vocabulary live.")},
]


# Everyday words -> the COCO-80 class that covers them. Used so the UI never claims YOLOv8 "cannot" find something
# it actually has a class for (e.g. necktie = tie). Rule: a word is listed as "no COCO class" only if neither the
# word itself nor a synonym below is one of YOLOv8's 80 class names. Ambiguous cases (coffee table ~ dining table)
# are mapped to the COCO class on purpose, to err on the side of NOT over-claiming.
COCO_SYNONYMS = {
    "people": "person", "man": "person", "woman": "person", "men": "person", "women": "person", "child": "person",
    "kid": "person", "boy": "person", "girl": "person", "human": "person", "pedestrian": "person", "guy": "person",
    "baby": "person", "player": "person", "rider": "person",
    "necktie": "tie", "bike": "bicycle", "bicycles": "bicycle", "motorbike": "motorcycle", "scooter": "motorcycle",
    "aeroplane": "airplane", "plane": "airplane", "aircraft": "airplane", "jet": "airplane",
    "sofa": "couch", "settee": "couch", "television": "tv", "tv monitor": "tv", "tvmonitor": "tv", "monitor": "tv",
    "screen": "tv", "mobile phone": "cell phone", "cellphone": "cell phone", "phone": "cell phone",
    "smartphone": "cell phone", "mobile": "cell phone", "iphone": "cell phone",
    "table": "dining table", "coffee table": "dining table", "desk": "dining table", "diningtable": "dining table",
    "computer mouse": "mouse", "notebook computer": "laptop", "computer keyboard": "keyboard",
    "remote control": "remote", "tv remote": "remote", "fridge": "refrigerator", "microwave oven": "microwave",
    "hair dryer": "hair drier", "hairdryer": "hair drier", "puppy": "dog", "kitten": "cat", "pony": "horse",
    "hydrant": "fire hydrant", "motorcar": "car", "automobile": "car", "taxi": "car", "lorry": "truck",
    "pickup truck": "truck", "ship": "boat", "sailboat": "boat", "yacht": "boat", "canoe": "boat", "kayak": "boat",
    "ski": "skis", "snowboards": "snowboard", "skate board": "skateboard", "surf board": "surfboard",
    "tennis racquet": "tennis racket", "racket": "tennis racket", "racquet": "tennis racket", "ball": "sports ball",
    "football": "sports ball", "soccer ball": "sports ball", "basketball": "sports ball", "baseball": "sports ball",
    "glass of wine": "wine glass", "mug": "cup", "coffee cup": "cup", "plant": "potted plant",
    "houseplant": "potted plant", "flower pot": "vase", "doughnut": "donut", "hotdog": "hot dog",
    "teddy": "teddy bear", "stuffed animal": "teddy bear", "traffic signal": "traffic light", "purse": "handbag",
    "bag": "handbag", "rucksack": "backpack", "suitcase bag": "suitcase", "luggage": "suitcase", "sheep": "sheep",
    "lamb": "sheep", "cattle": "cow", "bull": "cow", "toilet seat": "toilet", "sink basin": "sink",
    "parking sign": "parking meter", "stop signs": "stop sign", "sandwiches": "sandwich", "pizzas": "pizza",
    "bottles": "bottle", "chairs": "chair", "cars": "car", "dogs": "dog", "cats": "cat", "persons": "person",
}


def coco_class_for(word: str, coco_names: set[str]) -> str | None:
    """COCO-80 class name that covers `word` (exact, simple plural, or synonym), else None."""
    w = " ".join(word.lower().split())
    for cand in (w, w[:-1] if w.endswith("s") else w, COCO_SYNONYMS.get(w)):
        if cand and cand in coco_names:
            return cand
    return None


def parse_vocab(text: str | None) -> list[str]:
    """Comma/semicolon/newline separated names -> trimmed, de-duplicated (case-insensitive), non-empty list."""
    if not text:
        return []
    for sep in (";", "\n", "\r", "\t"):
        text = text.replace(sep, ",")
    out, seen = [], set()
    for raw in text.split(","):
        name = " ".join(raw.strip().split())  # collapse inner whitespace
        name = name.strip(" .'\"`")[:60]  # CLIP context is 77 tokens; keep labels short
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out[:MAX_CLASSES]


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    def as_dict(self) -> dict:
        return {"label": self.label, "conf": round(self.conf, 4), "box_xyxy": [round(v, 1) for v in self.box]}


@dataclass
class RunResult:
    detections: list[Detection]
    inference_ms: float  # model forward pass only (Ultralytics' own timer)
    predict_ms: float  # preprocess + forward + NMS (Ultralytics' three timers summed)
    encode_ms: float | None = None  # CLIP text encoding time if the vocabulary was (re-)encoded on this call
    extra: dict = field(default_factory=dict)


class ModelBank:
    """Loads models lazily, keeps them on the GPU, and re-encodes the vocabulary only when it changes."""

    def __init__(self) -> None:
        self.lock = threading.Lock()  # one GPU, one request at a time -> no races, predictable timings
        self.world: dict[str, YOLOWorld] = {}
        self.closed: dict[str, YOLO] = {}
        self.vocab: dict[str, tuple[str, ...]] = {}  # current encoded vocabulary per YOLO-World size
        self.clip = None  # shared CLIP text encoder

    # model loading
    def _world(self, size: str) -> YOLOWorld:
        if size not in self.world:
            m = YOLOWorld(str(WEIGHTS / f"yolov8{size.lower()}-worldv2.pt"))
            m.to("cuda" if DEVICE == 0 else "cpu")
            if self.clip is None:
                from ultralytics.nn.text_model import build_text_model

                self.clip = build_text_model("clip:ViT-B/32", device=torch.device("cuda" if DEVICE == 0 else "cpu"))
            m.model.clip_model = self.clip  # share one text encoder across S/M/L
            self.world[size] = m
            self.vocab[size] = ()
        return self.world[size]

    def _closed(self, size: str) -> YOLO:
        if size not in self.closed:
            self.closed[size] = YOLO(str(WEIGHTS / f"yolov8{size.lower()}.pt"))
        return self.closed[size]

    def warmup(self, sizes=SIZES, vocab: list[str] | None = None, log=print) -> None:
        """Load models and run dummy predictions so the first real click is fast."""
        vocab = vocab or parse_vocab(DEFAULT_VOCAB)
        dummy = np.full((640, 640, 3), 114, dtype=np.uint8)
        for s in sizes:
            t0 = time.perf_counter()
            self.run_world(dummy, vocab, s, conf=0.5)
            self.run_closed(dummy, s, conf=0.5)
            self.run_world(dummy, vocab, s, conf=0.5)
            self.run_closed(dummy, s, conf=0.5)
            log(f"  warmed up size {s} in {time.perf_counter() - t0:.1f}s")

    @staticmethod
    def _to_result(r, encode_ms=None) -> RunResult:
        names = r.names
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, confs, clss):
                dets.append(Detection(str(names[int(k)]), float(c), tuple(float(v) for v in b)))
        dets.sort(key=lambda d: -d.conf)
        sp = r.speed or {}
        return RunResult(
            detections=dets,
            inference_ms=float(sp.get("inference", float("nan"))),
            predict_ms=float(sum(sp.get(k, 0.0) for k in ("preprocess", "inference", "postprocess"))),
            encode_ms=encode_ms,
        )

    def run_world(self, img: np.ndarray, vocab: list[str], size: str = "S", conf: float = 0.15,
                  iou: float = 0.5, imgsz: int = 640) -> RunResult:
        with self.lock:
            m = self._world(size)
            encode_ms = None
            key = tuple(vocab)
            # prompt-then-detect: encode the words only when they change. Ultralytics keeps the embeddings as a
            # cached tensor, it does not fold them into the conv weights like the paper's re-parameterization.
            if self.vocab.get(size) != key:
                if DEVICE == 0:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                m.set_classes(list(vocab))
                if DEVICE == 0:
                    torch.cuda.synchronize()
                encode_ms = (time.perf_counter() - t0) * 1000
                self.vocab[size] = key
            r = m.predict(rgb_to_bgr(img), device=DEVICE, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            return self._to_result(r, encode_ms)

    def run_closed(self, img: np.ndarray, size: str = "S", conf: float = 0.25, iou: float = 0.5,
                   imgsz: int = 640) -> RunResult:
        with self.lock:
            m = self._closed(size)
            r = m.predict(rgb_to_bgr(img), device=DEVICE, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            return self._to_result(r)


# drawing (big boxes and labels, for the projector)
_PALETTE = [
    (255, 56, 56), (0, 194, 255), (255, 178, 29), (72, 249, 10), (207, 210, 49), (146, 204, 23),
    (61, 219, 134), (26, 147, 52), (0, 212, 187), (44, 153, 168), (52, 69, 147), (100, 115, 255),
    (132, 56, 255), (203, 56, 255), (255, 149, 200), (255, 157, 151),
]


def color_for(label: str) -> tuple[int, int, int]:
    h = 0
    for ch in label.lower():
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return _PALETTE[h % len(_PALETTE)]


_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def load_font(px: int) -> ImageFont.ImageFont:
    """A bold system font if available (Windows/macOS/Linux), else Pillow's built-in scalable font. No downloads."""
    if px in _FONT_CACHE:
        return _FONT_CACHE[px]
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    font = None
    for c in candidates:
        try:
            font = ImageFont.truetype(c, px)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=px)
        except TypeError:  # very old Pillow
            font = ImageFont.load_default()
    _FONT_CACHE[px] = font
    return font


def _overlap(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_w: int, px: int, min_px: int = 12):
    """Largest font <= px whose rendering of `text` fits in max_w pixels."""
    while px > min_px:
        f = load_font(px)
        tb = draw.textbbox((0, 0), text, font=f)
        if tb[2] - tb[0] <= max_w:
            return f
        px -= 2
    return load_font(min_px)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if cur and draw.textlength(trial, font=font) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def draw_detections(img: np.ndarray | Image.Image, dets: list[Detection], title: str | None = None,
                    subtitle: str | None = None, title_color=(30, 30, 30), min_long_side: int = 0) -> Image.Image:
    """Draw thick boxes + large labels (projector friendly).

    * boxes first, then all labels on top, so a big box never hides a small object's label
    * each label tries several spots (above the box, inside top, below, inside bottom) and takes the one that
      overlaps least with labels already placed -> readable even in crowded scenes (e.g. many cones)
    * `min_long_side` upsamples small images first so text stays crisp in slides
    """
    im = img.copy() if isinstance(img, Image.Image) else Image.fromarray(img)
    im = im.convert("RGB")
    s = 1.0
    if min_long_side and max(im.size) < min_long_side:
        s = min_long_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    W, H = im.size
    lw = max(3, round(max(W, H) / 280))
    fpx = max(16, round(max(W, H) / 40))
    font = load_font(fpx)
    pad = max(3, fpx // 6)
    d = ImageDraw.Draw(im)
    order = sorted(dets, key=lambda x: -x.conf)  # best first: gets the best label spot
    boxes = [tuple(v * s for v in det.box) for det in order]
    for det, (x1, y1, x2, y2) in sorted(zip(order, boxes), key=lambda t: t[0].conf):
        d.rectangle([x1, y1, x2, y2], outline=color_for(det.label), width=lw)
    placed: list[tuple[float, float, float, float]] = []
    for det, (x1, y1, x2, y2) in zip(order, boxes):
        col = color_for(det.label)
        text = f"{det.label} {det.conf:.2f}"
        tb = d.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0] + 2 * pad, tb[3] - tb[1] + 2 * pad
        tx = min(max(0, x1 - lw / 2), max(0, W - tw))
        cands = [y1 - th, y1, y2, y2 - th]  # above, inside-top, below, inside-bottom
        best, best_cost = None, None
        for i, ty in enumerate(cands):
            ty = min(max(0, ty), H - th)
            r = (tx, ty, tx + tw, ty + th)
            cost = sum(_overlap(r, p) for p in placed) + i * 0.01  # tie-break: prefer the natural order
            if best_cost is None or cost < best_cost:
                best, best_cost = r, cost
        placed.append(best)
        d.rectangle(best, fill=col)
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        d.text((best[0] + pad - tb[0], best[1] + pad - tb[1]), text,
               fill=(0, 0, 0) if lum > 140 else (255, 255, 255), font=font)
    if title is None:
        return im
    # header band with the model name (font shrinks to fit, subtitle wraps)
    margin = 14
    tfont = _fit_font(d, title, W - 2 * margin, max(22, round(W / 24)))
    sfont = load_font(max(15, round(W / 42)))
    tb = d.textbbox((0, 0), title, font=tfont)
    th = tb[3] - tb[1]
    sub_lines = _wrap(d, subtitle, sfont, W - 2 * margin) if subtitle else []
    sb = d.textbbox((0, 0), "Ag", font=sfont)
    sh = sb[3] - sb[1] + 8
    band = margin + th + 10 + len(sub_lines) * sh + margin
    canvas = Image.new("RGB", (W, H + band), (255, 255, 255))
    canvas.paste(im, (0, band))
    cd = ImageDraw.Draw(canvas)
    cd.text((margin - tb[0], margin - tb[1]), title, fill=title_color, font=tfont)
    y = margin + th + 10
    for line in sub_lines:
        cd.text((margin, y - sb[1]), line, fill=(80, 80, 80), font=sfont)
        y += sh
    return canvas


def side_by_side(left: Image.Image, right: Image.Image, gap: int = 16) -> Image.Image:
    h = max(left.height, right.height)
    out = Image.new("RGB", (left.width + right.width + gap, h), (255, 255, 255))
    out.paste(left, (0, h - left.height))  # bottom-align so the two photos line up
    out.paste(right, (left.width + gap, h - right.height))
    return out


def load_image(path_or_array) -> np.ndarray:
    """Return an RGB uint8 array. Handles grayscale/RGBA/palette images and EXIF rotation."""
    if isinstance(path_or_array, np.ndarray):
        arr = path_or_array
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return np.ascontiguousarray(arr.astype(np.uint8))
    from PIL import ImageOps

    im = Image.open(path_or_array)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im)


def rgb_to_bgr(arr: np.ndarray) -> np.ndarray:
    # Ultralytics treats numpy input as BGR (OpenCV convention), my arrays are RGB
    return np.ascontiguousarray(arr[..., ::-1])
