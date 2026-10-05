"""YOLO-World live demo: open-vocabulary detection next to a closed-set YOLOv8.

Run from the repo root (after python scripts/download_models.py):
    python demo/app.py
    python demo/app.py --sizes S    # only load the S models
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # sets the offline env vars before ultralytics/gradio are imported
from core import (
    DEFAULT_CONF, DEFAULT_CONF_CLOSED, DEFAULT_VOCAB, DEVICE_NAME, IMAGES, SAMPLES, SIZES, ModelBank,
    coco_class_for, draw_detections, load_image, parse_vocab,
)

import gradio as gr
import numpy as np
import torch
from PIL import Image

MAX_SIDE = 1280  # downscale big uploads (phone photos) before anything else

bank = ModelBank()
COCO80: set[str] = set()
LOADED: list[str] = list(SIZES)  # sizes offered in the UI; main() narrows it to --sizes


# helpers
def _fmt_ms(x: float) -> str:
    return "n/a" if x != x else f"{x:.1f} ms"  # NaN-safe


def _label_summary(dets) -> str:
    if not dets:
        return "<span class='none'>no detections above the threshold</span>"
    best: dict[str, float] = {}
    cnt = Counter(d.label for d in dets)
    for d in dets:
        best[d.label] = max(best.get(d.label, 0.0), d.conf)
    chips = []
    for lab, c in sorted(cnt.items(), key=lambda kv: -best[kv[0]]):
        n = f" x{c}" if c > 1 else ""
        chips.append(f"<span class='chip'><b>{html.escape(lab)}</b>{n} <small>{best[lab]:.2f}</small></span>")
    return " ".join(chips)


def _downscale(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s >= 1:
        return arr
    im = Image.fromarray(arr).resize((round(w * s), round(h * s)), Image.LANCZOS)
    return np.asarray(im)


def _error_box(msg: str) -> str:
    return f"<div class='err'>{msg}</div>"


_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


def _guarded(fn, *args, **kwargs):
    # run the model call on the worker thread so a stuck GPU gives a TimeoutError instead of hanging the UI
    return _EXEC.submit(fn, *args, **kwargs).result(timeout=core.GPU_CALL_TIMEOUT_S)


def detect(image, vocab_text, size, conf, conf_closed):
    """Returns (world_img, closed_img, world_md, closed_md, status_md)."""
    empty = (None, None, "", "", "")
    try:
        if image is None:
            return (*empty[:4], _error_box("Pick a sample image below or upload one first."))
        vocab = parse_vocab(vocab_text)
        img = _downscale(load_image(image))
        size = size if size in LOADED else LOADED[0]

        # the closed-set model runs no matter what the words are
        closed = _guarded(bank.run_closed, img, size, conf=float(conf_closed))
        closed_img = draw_detections(img, closed.detections, min_long_side=960)
        not_in_coco = [w for w in vocab if coco_class_for(w, COCO80) is None]
        mapped = [(w, coco_class_for(w, COCO80)) for w in vocab
                  if coco_class_for(w, COCO80) not in (None, w.lower())]
        closed_md = (
            f"<div class='time'><b>{_fmt_ms(closed.inference_ms)}</b> inference "
            f"<small>(forward pass only; {_fmt_ms(closed.predict_ms)} incl. pre/post-processing)</small></div>"
            f"<div class='labels'>{_label_summary(closed.detections)}</div>"
        )
        if not_in_coco:
            closed_md += (
                "<div class='miss'>No COCO class (or synonym) for: <b>"
                + html.escape(", ".join(not_in_coco)) + "</b>. YOLOv8 can never output these.</div>"
            )
        if mapped:
            closed_md += ("<div class='note'>Your words map to COCO classes: "
                          + html.escape(", ".join(f"{w} = {c}" for w, c in mapped)) + "</div>")

        if not vocab:
            status = _error_box(
                "The vocabulary is empty. Type some object names separated by commas, e.g. "
                "<i>person, guitar, helmet</i>. (YOLOv8 on the right still ran with its fixed 80 classes.)"
            )
            return None, closed_img, "", closed_md, status

        world = _guarded(bank.run_world, img, vocab, size, conf=float(conf))
        world_img = draw_detections(img, world.detections, min_long_side=960)
        if world.encode_ms is not None:
            enc = (f"New vocabulary: {len(vocab)} word{'s' if len(vocab) != 1 else ''} encoded by the CLIP text encoder once in "
                   f"<b>{world.encode_ms:.0f} ms</b>, cached, then reused for every image. (The paper goes further and folds the "
                   "embeddings into the detector weights; this demo only caches them.)")
        else:
            enc = (f"Vocabulary unchanged: re-using the cached text embeddings of {len(vocab)} words "
                   "(0 ms of text encoding this time; the cached embeddings are still used by the detection head).")
        world_md = (
            f"<div class='time'><b>{_fmt_ms(world.inference_ms)}</b> inference "
            f"<small>(forward pass only; {_fmt_ms(world.predict_ms)} incl. pre/post-processing)</small></div>"
            f"<div class='labels'>{_label_summary(world.detections)}</div>"
            f"<div class='enc'>{enc}</div>"
        )
        found = {d.label.lower() for d in world.detections}
        missing = [w for w in vocab if w.lower() not in found]
        status = (f"YOLO-World-{size} | {len(vocab)} word{'s' if len(vocab) != 1 else ''} | threshold {conf:.2f} | "
                  f"{len(world.detections)} boxes | device: {html.escape(DEVICE_NAME)}")
        if missing and len(missing) <= 15:
            status += f"<br><small>Not found at this threshold: {html.escape(', '.join(missing))}</small>"
        return world_img, closed_img, world_md, closed_md, f"<div class='ok'>{status}</div>"

    except concurrent.futures.TimeoutError:
        return (*empty[:4], _error_box(
            f"The model did not answer within {core.GPU_CALL_TIMEOUT_S:.0f} s. The GPU looks stuck (another program "
            "may be hogging it). Restart the demo; it falls back to the CPU automatically if the GPU stays stuck "
            "(or force it: set DEMO_DEVICE=cpu)."))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return (*empty[:4], _error_box("GPU ran out of memory. Try a smaller model size or fewer words."))
    except Exception as e:
        traceback.print_exc()
        return (*empty[:4], _error_box(f"Something went wrong: {html.escape(type(e).__name__)}: "
                                       f"{html.escape(str(e))[:400]}"))


# gradio UI
CSS = """
.gradio-container {max-width: 100% !important; font-size: 18px;}
h1 {font-size: 1.7em !important; margin: 0 !important;}
.subtitle {font-size: 1.15em; color: var(--body-text-color-subdued);}
#vocab textarea, #vocab input {font-size: 1.45em !important; font-weight: 600; line-height: 1.35;}
#vocab label span, .panel-title {font-size: 1.15em !important;}
#run {font-size: 1.4em !important; min-height: 64px;}
.panel-title {font-weight: 700; font-size: 1.35em !important; margin: 4px 0;}
.world-title {color: #0b7a3e;} .closed-title {color: #9a3412;}
.time {font-size: 1.15em; margin: 4px 0;}
.labels {margin: 6px 0; line-height: 2.0;}
.chip {display: inline-block; padding: 2px 10px; margin: 2px; border-radius: 14px; font-size: 1.1em;
       background: var(--block-background-fill); border: 1px solid var(--border-color-primary);}
.none {color: var(--body-text-color-subdued); font-style: italic;}
.miss {margin-top: 6px; padding: 6px 10px; border-radius: 8px; background: rgba(234, 88, 12, .12);}
.enc {margin-top: 6px; padding: 6px 10px; border-radius: 8px; background: rgba(22, 163, 74, .12);}
.err {font-size: 1.2em; padding: 10px 14px; border-radius: 8px; background: rgba(220, 38, 38, .15);}
.ok {font-size: 1.0em; padding: 6px 10px;}
.note {font-size: 0.95em; color: var(--body-text-color-subdued);}
"""


def build_ui() -> gr.Blocks:
    samples = [[str(IMAGES / s["file"]), s["vocab"], s.get("conf", DEFAULT_CONF)]
               for s in SAMPLES if (IMAGES / s["file"]).exists()]
    first_img = samples[0][0] if samples else None

    with gr.Blocks(title="YOLO-World live demo") as demo:
        gr.Markdown("# YOLO-World: detect anything you can name, in real time")
        gr.Markdown(
            "<div class='subtitle'>Type object names (comma separated) and press <b>Enter</b>. "
            "Left: YOLO-World (open vocabulary, CVPR 2024). Right: YOLOv8 of the same size, "
            "which only knows the 80 fixed COCO classes.</div>"
        )
        with gr.Row():
            with gr.Column(scale=5):
                vocab = gr.Textbox(value=samples[0][1] if samples else DEFAULT_VOCAB, lines=1, max_lines=4,
                                   label="Vocabulary: what should YOLO-World look for?", elem_id="vocab",
                                   placeholder="e.g. person, guitar, helmet, sunglasses, flag")
                status = gr.HTML()
            with gr.Column(scale=2, min_width=260):
                size = gr.Radio(LOADED, value=LOADED[0], label="Model size (both models)")
                conf = gr.Slider(0.01, 0.9, value=samples[0][2] if samples else DEFAULT_CONF, step=0.01,
                                 label="YOLO-World confidence threshold")
                run = gr.Button("Detect", variant="primary", elem_id="run")
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                image = gr.Image(value=first_img, type="numpy", label="Input image (upload, paste or pick below)",
                                 sources=["upload", "clipboard", "webcam"], height=230)
                with gr.Accordion("Advanced", open=False):
                    conf_closed = gr.Slider(0.01, 0.9, value=DEFAULT_CONF_CLOSED, step=0.01,
                                            label="YOLOv8 confidence threshold (Ultralytics default 0.25)")
            with gr.Column(scale=2):
                gr.HTML("<div class='panel-title world-title'>YOLO-World (open vocabulary: your words)</div>")
                out_world = gr.Image(label="YOLO-World", show_label=False, format="png", height=440,
                                     buttons=["fullscreen", "download"], interactive=False)
                md_world = gr.HTML()
            with gr.Column(scale=2):
                gr.HTML("<div class='panel-title closed-title'>YOLOv8 (closed set: fixed 80 COCO classes)</div>")
                out_closed = gr.Image(label="YOLOv8", show_label=False, format="png", height=440,
                                      buttons=["fullscreen", "download"], interactive=False)
                md_closed = gr.HTML()

        inputs = [image, vocab, size, conf, conf_closed]
        outputs = [out_world, out_closed, md_world, md_closed, status]
        if samples:
            # clicking a sample fills image + vocabulary + its tuned threshold; image.change then runs detection
            gr.Examples(samples, inputs=[image, vocab, conf], label="Sample images (click one)",
                        cache_examples=False, examples_per_page=8)
        gr.Markdown(
            f"<div class='note'>Times are measured on {html.escape(core.DEVICE_NAME)} after warm-up and cover the model's forward pass "
            "only (Ultralytics' own timer); the value in brackets adds image pre-processing and NMS. "
            "YOLO-World's text encoder (CLIP ViT-B/32) runs only when the vocabulary changes: the words are "
            "encoded once and the embeddings are cached and reused (the paper's \"prompt-then-detect\" idea; the paper "
            "additionally re-parameterizes them into the weights, which this demo does not do). "
            "Sample images: COCO val2017 (Flickr, CC BY / CC BY-SA), "
            "see demo/images/ATTRIBUTION.md.</div>"
        )

        # triggers: button, Enter in the textbox, size change, slider release, new image
        run.click(detect, inputs, outputs, concurrency_limit=1)
        vocab.submit(detect, inputs, outputs, concurrency_limit=1)
        size.change(detect, inputs, outputs, concurrency_limit=1)
        conf.release(detect, inputs, outputs, concurrency_limit=1)
        conf_closed.release(detect, inputs, outputs, concurrency_limit=1)
        image.change(lambda *a: detect(*a) if a[0] is not None else (None, None, "", "", ""),
                     inputs, outputs, concurrency_limit=1, show_progress="minimal")
        demo.load(detect, inputs, outputs)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO-World vs YOLOv8 live demo")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link (needs internet)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from another device on the LAN")
    ap.add_argument("--sizes", default="SML", help="model sizes to load and offer in the UI, e.g. S or SML")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args()

    global COCO80
    print(f"YOLO-World demo | device: {DEVICE_NAME} | torch {torch.__version__}")
    sizes = [s for s in SIZES if s in args.sizes.upper()] or ["S"]
    LOADED[:] = sizes
    t0 = time.time()
    print(f"Loading + warming up models {sizes} (one-time, before the UI opens)...")
    bank.warmup(sizes)
    COCO80 = {str(n).lower() for n in bank._closed(sizes[0]).names.values()}
    print(f"Ready in {time.time() - t0:.1f}s")

    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=16)
    demo.launch(
        share=args.share, server_name=args.host, server_port=args.port, inbrowser=not args.no_browser,
        show_error=True, ssr_mode=False, footer_links=[],
        theme=gr.themes.Soft(font=["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
                             font_mono=["Consolas", "Menlo", "monospace"], text_size="lg"),
        css=CSS,
    )


if __name__ == "__main__":
    main()
