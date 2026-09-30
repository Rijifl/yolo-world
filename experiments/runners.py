"""Wrappers that give every detector the same interface.

    set_vocab(names)    encode the text vocabulary once (the caller times this separately)
    predict(img, conf)  -> (boxes_xyxy [N,4] in original-image pixels, scores [N], class_idx [N]) as numpy
    info()              dict with params, input resolution, precision
predict() covers everything after the image is decoded in memory: preprocessing (resize/letterbox/normalise,
host->GPU copy), forward pass, and post-processing (NMS or top-k).
"""
from __future__ import annotations

import numpy as np
import torch

MAX_DET = 100


def n_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# Ultralytics models
def _box_only_predictor():
    """DetectionPredictor that accepts a *-seg model's (preds, protos) output and ignores the mask protos,
    so YOLOE-seg checkpoints are evaluated/timed as box detectors (mask coefficients are dropped after NMS)."""
    from ultralytics.models.yolo.detect import DetectionPredictor

    class BoxOnlyPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            p = preds[0] if isinstance(preds, (list, tuple)) else preds
            p = p[0] if isinstance(p, (list, tuple)) else p
            res = super().postprocess(p, img, orig_imgs, **kwargs)
            return res

    return BoxOnlyPredictor


class UltralyticsRunner:
    """YOLOv8 (closed set), YOLO-World v2 and YOLOE via the Ultralytics predict() API.

    kind = 'yolo' | 'world' | 'yoloe'. Images must be BGR uint8 numpy arrays (Ultralytics convention).
    YOLOE checkpoints are *-seg models; they are run through the plain DetectionPredictor so only boxes are
    decoded (the mask branch still runs in the forward pass, but mask decoding/upsampling is skipped).
    """

    def __init__(self, weight: str, kind: str, device: str = "cuda:0", half: bool = True, imgsz: int = 640,
                 iou: float = 0.7):
        from ultralytics import YOLO, YOLOE

        self.weight, self.kind, self.device, self.half, self.imgsz, self.iou = weight, kind, device, half, imgsz, iou
        self.model = YOLOE(weight) if kind == "yoloe" else YOLO(weight)
        self.model.to(device)
        self.names = None

    def set_vocab(self, names: list[str]) -> None:
        """Offline text encoding ("prompt-then-detect")."""
        if self.kind == "yolo":
            self.names = list(self.model.names.values())
            return
        if self.kind == "world":
            self.model.set_classes(list(names))  # CLIP ViT-B/32 text encoder, cached after first call
        else:  # yoloe: MobileCLIP text encoder -> RepRTA aux head -> embeddings, then head is fused at predict time
            pe = self.model.model.get_text_pe(list(names), cache_clip_model=True)
            self.model.set_classes(list(names), pe)
        self.names = list(names)

    def encode_text_only(self, names: list[str]):
        """Just the text-encoder part (no model state change); used for timing."""
        if self.kind == "world":
            return self.model.model.get_text_pe(list(names), cache_clip_model=True)
        if self.kind == "yoloe":
            return self.model.model.get_text_pe(list(names), cache_clip_model=True)
        return None

    def predict(self, img_bgr: np.ndarray, conf: float = 0.001, max_det: int = MAX_DET):
        kw = dict(imgsz=self.imgsz, conf=conf, iou=self.iou, max_det=max_det, quantize=16 if self.half else 32, device=self.device,
                  verbose=False)
        if self.kind == "yoloe":
            from ultralytics.engine.model import Model

            # Base-class predict (bypasses YOLOE.predict's visual-prompt path) with a boxes-only predictor.
            r = Model.predict(self.model, img_bgr, predictor=_box_only_predictor(), **kw)[0]
        else:
            r = self.model.predict(img_bgr, **kw)[0]
        b = r.boxes
        self.last_speed = r.speed
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)

    def info(self) -> dict:
        det = n_params(self.model.model.model)
        text = None
        cm = getattr(self.model.model, "clip_model", None)
        if self.kind == "world" and cm is not None:
            # Ultralytics wraps the FULL OpenAI CLIP model; count only the text side (total - vision tower).
            text = n_params(cm.model) - n_params(cm.model.visual)
        elif self.kind == "yoloe" and cm is not None:
            enc = getattr(cm, "encoder", None)
            te = getattr(enc, "text_encoder", None) if enc is not None else None
            # mobileclip_blt.ts exposes image_encoder + text_encoder; mobileclip2_b.ts is a frozen TorchScript
            # graph that exposes no parameters -> reported as None.
            text = n_params(te) if te is not None else None
            if text == 0:
                text = None
        return {"params_detector_M": round(det / 1e6, 2), "params_text_encoder_M": None if text is None else round(text / 1e6, 2),
                "input_res": f"{self.imgsz} (letterbox, rect)", "precision": "fp16" if self.half else "fp32"}


class OWLv2Runner:
    """google/owlv2-base-patch16-ensemble via Hugging Face transformers.

    Two inference paths (identical outputs, verified in compare.py):
      * predict():        text embeddings of the vocabulary are computed ONCE in set_vocab() and reused
                          (image_embedder -> class_predictor -> box_predictor). This is the fastest fair setup.
      * predict_naive():  the standard HF call model(input_ids, pixel_values) which re-encodes all text queries
                          for every image.
    Post-processing: sigmoid class logits over all (box, query) pairs, top-100 pairs per image (DETR/OWL style,
    no NMS). Boxes are relative to the 960x960 zero-padded square, so they are scaled by max(H, W).
    """

    CKPT = "google/owlv2-base-patch16-ensemble"

    def __init__(self, device: str = "cuda:0", half: bool = True):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device, self.half = device, half
        self.dtype = torch.float16 if half else torch.float32
        self.processor = Owlv2Processor.from_pretrained(self.CKPT)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.CKPT, dtype=self.dtype).to(device).eval()
        self.names = None

    @torch.inference_mode()
    def _encode(self, names):
        tok = self.processor.tokenizer(list(names), padding="max_length", max_length=16, truncation=True,
                                       return_tensors="pt").to(self.device)
        out = self.model.owlv2.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        return emb, tok

    @torch.inference_mode()
    def set_vocab(self, names):
        self.names = list(names)
        emb, tok = self._encode(names)
        self.query_embeds = emb[None]  # [1, Q, D]
        self.query_mask = (tok["input_ids"][:, 0] > 0)[None]
        self.tok = tok

    def encode_text_only(self, names):
        return self._encode(names)[0]

    def _pixels(self, img_rgb):
        px = self.processor.image_processor(images=img_rgb, return_tensors="pt")["pixel_values"]
        return px.to(self.device, self.dtype, non_blocking=True)

    def _post(self, logits, boxes, h, w, conf, max_det):
        scores = torch.sigmoid(logits[0].float())  # [P, Q]
        P, Q = scores.shape
        s, idx = scores.flatten().topk(min(max_det, P * Q))
        keep = s > conf
        s, idx = s[keep], idx[keep]
        bi, ci = idx // Q, idx % Q
        b = boxes[0].float()[bi]
        b = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        b = (b * max(h, w)).clamp(min=0)
        b[:, [0, 2]] = b[:, [0, 2]].clamp(max=w)
        b[:, [1, 3]] = b[:, [1, 3]].clamp(max=h)
        return b.cpu().numpy(), s.cpu().numpy(), ci.cpu().numpy().astype(int)

    @torch.inference_mode()
    def predict(self, img_rgb, conf: float = 0.001, max_det: int = MAX_DET):
        h, w = img_rgb.shape[:2]
        px = self._pixels(img_rgb)
        feat_map, _ = self.model.image_embedder(pixel_values=px)
        B, hh, ww, D = feat_map.shape
        feats = feat_map.reshape(B, hh * ww, D)
        logits, _ = self.model.class_predictor(feats, self.query_embeds, self.query_mask)
        boxes = self.model.box_predictor(feats, feat_map)
        return self._post(logits, boxes, h, w, conf, max_det)

    @torch.inference_mode()
    def predict_naive(self, img_rgb, conf: float = 0.001, max_det: int = MAX_DET):
        """Standard HF forward: the text encoder runs on all queries for every image."""
        h, w = img_rgb.shape[:2]
        px = self._pixels(img_rgb)
        tok = self.processor.tokenizer(self.names, padding="max_length", max_length=16, truncation=True,
                                       return_tensors="pt").to(self.device)
        out = self.model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], pixel_values=px)
        return self._post(out.logits, out.pred_boxes, h, w, conf, max_det)

    def info(self):
        text = n_params(self.model.owlv2.text_model) + n_params(self.model.owlv2.text_projection)
        return {"params_detector_M": round((n_params(self.model) - text) / 1e6, 2),
                "params_text_encoder_M": round(text / 1e6, 2),
                "input_res": "960x960 (pad to square, resize)", "precision": "fp16" if self.half else "fp32"}


class GroundingDINORunner:
    """IDEA-Research/grounding-dino-tiny via Hugging Face transformers.

    Follows the official GroundingDINO COCO evaluation (demo/test_ap_on_coco.py): ONE caption with all 80 class
    names joined by " . " (195 BERT tokens < the 256-token limit, so no chunking is needed). Per-class score for
    each of the 900 queries = mean over that class's tokens of sigmoid(token logits) (official PostProcessCocoGrounding
    uses a row-normalised positive map, i.e. the same mean), then top-100 (query, class) pairs per image, no NMS.
    Text and image are fused inside the encoder/decoder, so the text branch necessarily runs on every image.
    """

    CKPT = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, device: str = "cuda:0", half: bool = True):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device, self.half = device, half
        self.dtype = torch.float16 if half else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.CKPT)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.CKPT, dtype=self.dtype).to(device).eval()
        self.names = None

    def set_vocab(self, names):
        self.names = list(names)
        caption = " . ".join(n.lower().strip() for n in names) + " ."
        tok = self.processor.tokenizer(caption, return_offsets_mapping=True, return_tensors="pt")
        assert tok["input_ids"].shape[1] <= self.model.config.max_text_len, "caption exceeds Grounding DINO token limit"
        offsets = tok.pop("offset_mapping")[0].tolist()
        # positive map: class c <-> tokens whose char span lies inside the class-name span in the caption
        pos = torch.zeros(len(names), self.model.config.max_text_len)
        start = 0
        for c, n in enumerate(names):
            n = n.lower().strip()
            s = caption.index(n, start)
            e = s + len(n)
            start = e
            for t, (a, b) in enumerate(offsets):
                if b > a and a >= s and b <= e:
                    pos[c, t] = 1
        assert (pos.sum(1) > 0).all()
        self.pos = (pos / pos.sum(1, keepdim=True)).to(self.device)  # row-normalised
        self.caption = caption
        self.text_inputs = {k: v.to(self.device) for k, v in tok.items()}
        self.n_tokens = int(tok["input_ids"].shape[1])

    @torch.inference_mode()
    def encode_text_only(self, names):
        """BERT text backbone only (informational: its output is fused with the image in every forward)."""
        return self.model.model.text_backbone(input_ids=self.text_inputs["input_ids"],
                                              attention_mask=self.text_inputs["attention_mask"])

    @torch.inference_mode()
    def predict(self, img_rgb, conf: float = 0.001, max_det: int = MAX_DET):
        h, w = img_rgb.shape[:2]
        im = self.processor.image_processor(images=img_rgb, return_tensors="pt")
        out = self.model(pixel_values=im["pixel_values"].to(self.device, self.dtype, non_blocking=True),
                         pixel_mask=im["pixel_mask"].to(self.device, non_blocking=True), **self.text_inputs)
        prob = torch.sigmoid(out.logits[0].float())  # [900, 256]
        L = prob.shape[1]
        cls = prob @ self.pos[:, :L].T  # [900, C]
        Qn, C = cls.shape
        s, idx = cls.flatten().topk(min(max_det, Qn * C))
        keep = s > conf
        s, idx = s[keep], idx[keep]
        qi, ci = idx // C, idx % C
        b = out.pred_boxes[0].float()[qi]  # cxcywh normalised to the (unpadded) image
        b = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        b = b * torch.tensor([w, h, w, h], device=b.device)
        return b.cpu().numpy(), s.cpu().numpy(), ci.cpu().numpy().astype(int)

    def info(self):
        text = n_params(self.model.model.text_backbone)
        sz = self.processor.image_processor.size
        return {"params_detector_M": round((n_params(self.model) - text) / 1e6, 2),
                "params_text_encoder_M": round(text / 1e6, 2),
                "input_res": f"shortest edge {sz.get('shortest_edge')}, longest <= {sz.get('longest_edge')}",
                "precision": "fp16" if self.half else "fp32"}
