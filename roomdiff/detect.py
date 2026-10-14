"""
detect.py - open vocabulary detection, masks and appearance embeddings

Grounding DINO gives labelled boxes, SAM turns them into masks and CLIP embeds
each crop so an object can be recognised again in the other video. All three
run through Hugging Face transformers.
"""

import json

import cv2
import numpy as np
from PIL import Image

from .objects import Detection, canonical
from .visibility import SAME, DepthCache, ray_check

SKIP = {'wall', 'floor', 'ceiling', 'door', 'window', 'rug', 'carpet', 'curtain', 'room'}


def _iou(a, b):
    """Box IoU"""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


class Models:
    """Grounding DINO, SAM and CLIP"""
    def __init__(self, cfg, device=None, need=('det', 'sam', 'clip')):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel,
                                  CLIPProcessor, SamModel, SamProcessor)
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        if 'det' in need:
            self.det_proc = AutoProcessor.from_pretrained(cfg.detector_model)
            self.det = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.detector_model).to(self.device).eval()
        if 'sam' in need:
            self.sam_proc = SamProcessor.from_pretrained(cfg.sam_model)
            self.sam = SamModel.from_pretrained(cfg.sam_model).to(self.device).eval()
        if 'clip' in need:
            self.clip_proc = CLIPProcessor.from_pretrained(cfg.clip_model)
            self.clip = CLIPModel.from_pretrained(cfg.clip_model).to(self.device).eval()

    # detection
    def _prompt(self, img, classes):
        """Labelled boxes for one image and one class list, a few classes per prompt"""
        torch, cfg = self.torch, self.cfg
        found = []
        for k in range(0, len(classes), cfg.classes_per_prompt):
            chunk = classes[k:k + cfg.classes_per_prompt]
            text = '. '.join(chunk) + '.'
            inputs = self.det_proc(images=img, text=text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                out = self.det(**inputs)
            kw = dict(text_threshold=cfg.text_threshold, target_sizes=[img.size[::-1]])
            try:
                r = self.det_proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, threshold=cfg.box_threshold, **kw)[0]
            except TypeError:   # transformers < 4.51
                r = self.det_proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, box_threshold=cfg.box_threshold, **kw)[0]
            labels = r.get('text_labels', r.get('labels'))
            for box, score, lab in zip(r['boxes'].cpu().numpy(), r['scores'].cpu().numpy(), labels):
                lab = canonical(str(lab), chunk)
                if not lab or lab in SKIP:
                    continue
                found.append((lab, float(score), [float(x) for x in box]))
        return found

    def detect(self, img):
        """Labelled boxes for one image: the main list, then the extra list where it found nothing"""
        W, H = img.size

        def nms(found, keep):
            # drop near-full-frame boxes and specks, then class-agnostic NMS
            for lab, sc, b in sorted(found, key=lambda t: -t[1]):
                area = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)
                if area > 0.7 or area < 0.0015:
                    continue
                if any(_iou(b, k[2]) > 0.6 for k in keep):
                    continue
                keep.append((lab, sc, b))
            return keep

        keep = nms(self._prompt(img, [c for c in self.cfg.classes if c not in SKIP]), [])
        extra = [c for c in self.cfg.extra_classes if c not in SKIP]
        if extra:
            keep = nms(self._prompt(img, extra), keep)
        return keep

    def segment(self, img, boxes):
        """SAM masks for the boxes"""
        if not boxes:
            return []
        torch = self.torch
        inputs = self.sam_proc(img, input_boxes=[boxes], return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.sam(**inputs, multimask_output=False)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())[0]
        return [m[0].numpy().astype(bool) for m in masks]

    def embed_images(self, crops):
        """Normalised CLIP image embeddings"""
        if not crops:
            return np.zeros((0, 512), np.float32)
        torch = self.torch
        inputs = self.clip_proc(images=crops, return_tensors='pt').to(self.device)
        with torch.no_grad():
            f = self.clip.get_image_features(**inputs)
        f = f if torch.is_tensor(f) else f.pooler_output       # transformers v5 returns an output object
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        return f.cpu().numpy()

    def embed_texts(self, texts):
        """Normalised CLIP text embeddings"""
        torch = self.torch
        inputs = self.clip_proc(text=texts, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            f = self.clip.get_text_features(**inputs)
        f = f if torch.is_tensor(f) else f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()


def _crop(img, box, pad=0.1):
    """Crop around a box with some padding"""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return img.crop((max(0, x0 - pad * w), max(0, y0 - pad * h),
                     min(img.width, x1 + pad * w), min(img.height, y1 + pad * h)))


def detect_scene(scene, cfg):
    """Detections, masks and embeddings for every keyframe"""
    m = Models(cfg)
    H, W = scene.hw
    dets = []
    for i in range(scene.n):
        img = Image.open(scene.hires_paths[i]).convert('RGB') if scene.hires_paths \
            else Image.fromarray(scene.images[i])
        sx, sy = W / img.width, H / img.height
        found = m.detect(img)
        masks = m.segment(img, [b for _, _, b in found])
        embs = m.embed_images([_crop(img, b) for _, _, b in found])
        for (lab, sc, b), mk, e in zip(found, masks, embs):
            small = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            if small.sum() < 30:
                continue
            dets.append(Detection(frame=i, label=lab, score=sc,
                                  box=(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy), mask=small, embedding=e))
        print(f'[roomdiff]   frame {i + 1}/{scene.n}: ' + ', '.join(sorted({d[0] for d in found})), flush=True)
    del m
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return dets


def save_detections(dets, path):
    """Saves detections to npz"""
    meta = [{'frame': d.frame, 'label': d.label, 'score': d.score, 'box': list(map(float, d.box))} for d in dets]
    masks = np.stack([np.packbits(d.mask) for d in dets]) if dets else np.zeros((0, 1), np.uint8)
    shape = np.array(dets[0].mask.shape if dets else (0, 0))
    embs = np.stack([d.embedding for d in dets]) if dets and dets[0].embedding is not None else np.zeros((0, 1))
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), masks=masks, shape=shape, embs=embs)


def load_detections(path):
    """Loads detections from npz"""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    H, W = z['shape']
    out = []
    for k, m in enumerate(meta):
        mask = np.unpackbits(z['masks'][k])[: H * W].reshape(H, W).astype(bool)
        emb = z['embs'][k] if len(z['embs']) == len(meta) else None
        out.append(Detection(m['frame'], m['label'], m['score'], tuple(m['box']), mask, emb))
    return out


class ClipRegionNamer:
    """Best guess name for an unidentified object, CLIP on its best view"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.m = Models(cfg, need=('clip',))
        allc = list(cfg.classes) + list(cfg.extra_classes)
        self.vocab = sorted({canonical(c, allc) for c in allc} - SKIP)
        self.text = self.m.embed_texts([f'a photo of a {c}' for c in self.vocab])

    def __call__(self, scene, obj):
        frames = scene.frames_of(obj.session)
        pv = ray_check(obj.points, scene, frames, cache=DepthCache(scene))
        k = int(np.argmax((pv == SAME).sum(1)))
        i = int(frames[k])
        vis = obj.points[pv[k] == SAME]
        if len(vis) < 10:
            return None
        u, v, _ = scene.project(vis, i)
        img = Image.open(scene.hires_paths[i]).convert('RGB') if scene.hires_paths else Image.fromarray(scene.images[i])
        H, W = scene.hw
        f = img.width / W
        box = [np.percentile(u, 2) * f, np.percentile(v, 2) * f, np.percentile(u, 98) * f, np.percentile(v, 98) * f]
        e = self.m.embed_images([_crop(img, box, 0.15)])[0]
        logits = 100 * self.text @ e
        p = np.exp(logits - logits.max())
        p /= p.sum()
        j = int(np.argmax(p))
        return f'a {self.vocab[j]}' if p[j] >= 0.35 else None
