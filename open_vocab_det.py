import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam2Model,
    Sam2Processor,
)


def parse_queries(query_str: str) -> List[str]:
    raw_items = [x.strip().lower() for x in query_str.split(",") if x.strip()]
    if not raw_items:
        raise ValueError("No valid text queries were provided.")

    phrases = []
    for item in raw_items:
        if item.startswith(("a ", "an ", "the ")):
            phrases.append(item)
        else:
            phrases.append(f"a {item}")
    return phrases


def build_grounding_text(phrases: List[str]) -> str:
    return ". ".join([p.strip().lower().rstrip(".") for p in phrases]) + "."


def load_grounding_dino(model_name: str, device: str, use_fast: bool = False):
    processor = AutoProcessor.from_pretrained(model_name, use_fast=use_fast)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def get_sam2_dtype(device: str):
    if not device.startswith("cuda"):
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_sam2(model_name: str, device: str):
    dtype = get_sam2_dtype(device)
    processor = Sam2Processor.from_pretrained(model_name)
    model = Sam2Model.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return processor, model, dtype


@torch.inference_mode()
def run_grounding_dino(
    image: Image.Image,
    text: str,
    processor,
    model,
    threshold: float,
    text_threshold: float,
    device: str,
):
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (H, W)
    )
    return results[0]


@torch.inference_mode()
def run_sam2_with_boxes(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    processor,
    model,
    device: str,
):
    """
    boxes_xyxy: shape [N, 4]
    SAM2 box 输入格式:
      单张图 + 多个 box -> [ [ [x1,y1,x2,y2], [x1,y1,x2,y2], ... ] ]
    """
    if len(boxes_xyxy) == 0:
        return np.zeros((0, image.size[1], image.size[0]), dtype=bool), np.zeros((0,), dtype=float)

    input_boxes = [boxes_xyxy.tolist()]

    inputs = processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs, multimask_output=False)

    # 官方文档展示的做法：post_process_masks + original_sizes
    all_masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )

    masks = all_masks[0]  # 单张图
    # 常见形状可能是 [num_objects, 1, H, W] 或 [num_objects, H, W]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 4 and masks.shape[0] == 1:
        masks = masks[0]

    masks = masks.numpy() > 0

    # iou_scores 可能是 [1, num_objects, 1] / [1, num_objects] / 类似结构
    iou_scores = outputs.iou_scores.detach().float().cpu().numpy()
    iou_scores = np.array(iou_scores).reshape(-1)

    # 对齐长度
    num_obj = masks.shape[0]
    if len(iou_scores) < num_obj:
        padded = np.zeros((num_obj,), dtype=float)
        padded[: len(iou_scores)] = iou_scores
        iou_scores = padded
    else:
        iou_scores = iou_scores[:num_obj]

    return masks, iou_scores


def load_font(font_size: int = 18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def compute_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_overlay(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
    masks: np.ndarray,
):
    image_np = np.array(image).copy()
    h, w = image_np.shape[:2]

    overlay = np.zeros((h, w, 4), dtype=np.float32)

    # 几组固定颜色，避免随机带来展示不稳定
    color_bank = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.5, 1.0),
        (1.0, 0.5, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.7, 1.0, 0.0),
        (0.8, 0.2, 0.2),
    ]

    for i, mask in enumerate(masks):
        color = color_bank[i % len(color_bank)]
        overlay[..., 0] += mask.astype(np.float32) * color[0] * 0.35
        overlay[..., 1] += mask.astype(np.float32) * color[1] * 0.35
        overlay[..., 2] += mask.astype(np.float32) * color[2] * 0.35
        overlay[..., 3] = np.maximum(overlay[..., 3], mask.astype(np.float32) * 0.45)

    overlay[..., :3] = np.clip(overlay[..., :3], 0.0, 1.0)

    base = Image.fromarray(image_np)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8), mode="RGBA")
    composed = Image.alpha_composite(base.convert("RGBA"), overlay_img).convert("RGB")

    draw = ImageDraw.Draw(composed)
    font = load_font(18)

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = tuple(int(255 * c) for c in color_bank[i % len(color_bank)])

        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

        text = f"{label}: {float(score):.3f}"
        try:
            bbox = draw.textbbox((x1, y1), text, font=font)
            tx1, ty1, tx2, ty2 = bbox
            text_w, text_h = tx2 - tx1, ty2 - ty1
        except Exception:
            text_w = 8 * len(text)
            text_h = 18

        rect_y1 = max(0, y1 - text_h - 6)
        rect_y2 = rect_y1 + text_h + 4
        rect_x1 = x1
        rect_x2 = x1 + text_w + 8

        draw.rectangle([(rect_x1, rect_y1), (rect_x2, rect_y2)], fill=color)
        draw.text((rect_x1 + 4, rect_y1 + 2), text, fill="white", font=font)

    return composed


def save_masks(mask_dir: str, base_name: str, masks: np.ndarray, labels: List[str]):
    os.makedirs(mask_dir, exist_ok=True)
    saved_paths = []

    for i, (mask, label) in enumerate(zip(masks, labels)):
        mask_img = (mask.astype(np.uint8) * 255)
        safe_label = label.replace(" ", "_").replace("/", "_")
        path = os.path.join(mask_dir, f"{base_name}_mask_{i:02d}_{safe_label}.png")
        Image.fromarray(mask_img).save(path)
        saved_paths.append(path)

    return saved_paths


def save_json(
    out_path: str,
    boxes: np.ndarray,
    det_scores: np.ndarray,
    labels: List[str],
    sam_scores: np.ndarray,
    masks: np.ndarray,
):
    records = []
    for i, (box, det_score, label, sam_score, mask) in enumerate(
        zip(boxes, det_scores, labels, sam_scores, masks)
    ):
        x1, y1, x2, y2 = compute_mask_bbox(mask)
        records.append(
            {
                "index": i,
                "label": str(label),
                "detection_score": float(det_score),
                "sam2_score": float(sam_score),
                "box_xyxy": [float(x) for x in box],
                "mask_bbox_xyxy": [x1, y1, x2, y2],
                "mask_area": int(mask.sum()),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--text_queries", type=str, required=True)
    parser.add_argument("--gdino_model", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam2_model", type=str, default="facebook/sam2.1-hiera-base-plus")
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--max_detections", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--save_masks", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    image = Image.open(args.image_path).convert("RGB")
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]

    phrases = parse_queries(args.text_queries)
    text = build_grounding_text(phrases)

    print(f"[INFO] image: {args.image_path}")
    print(f"[INFO] text prompt: {text}")
    print(f"[INFO] Grounding DINO: {args.gdino_model}")
    print(f"[INFO] SAM2: {args.sam2_model}")
    print(f"[INFO] device: {args.device}")

    gdino_processor, gdino_model = load_grounding_dino(
        model_name=args.gdino_model,
        device=args.device,
        use_fast=False,
    )

    det_result = run_grounding_dino(
        image=image,
        text=text,
        processor=gdino_processor,
        model=gdino_model,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )

    boxes = det_result["boxes"].detach().cpu().numpy() if len(det_result["boxes"]) > 0 else np.zeros((0, 4))
    det_scores = det_result["scores"].detach().cpu().numpy() if len(det_result["scores"]) > 0 else np.zeros((0,))
    labels = list(det_result["labels"])

    if len(labels) == 0:
        print("[INFO] No detections found.")
        if args.save_json:
            json_path = os.path.join(args.out_dir, f"{base_name}_open_vocab_seg.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            print(f"[SAVED] json: {json_path}")
        return

    # 按检测分数排序并截断
    order = np.argsort(-det_scores)
    order = order[: args.max_detections]
    boxes = boxes[order]
    det_scores = det_scores[order]
    labels = [labels[i] for i in order]

    print(f"[INFO] detections kept: {len(labels)}")
    for i, (box, score, label) in enumerate(zip(boxes, det_scores, labels)):
        print(f"[DET {i:02d}] label={label:<20} score={float(score):.4f} box={[round(float(x), 1) for x in box]}")

    sam2_processor, sam2_model, _ = load_sam2(
        model_name=args.sam2_model,
        device=args.device,
    )

    masks, sam_scores = run_sam2_with_boxes(
        image=image,
        boxes_xyxy=boxes,
        processor=sam2_processor,
        model=sam2_model,
        device=args.device,
    )

    print(f"[INFO] masks predicted: {len(masks)}")
    for i, (label, sam_score, mask) in enumerate(zip(labels, sam_scores, masks)):
        print(f"[SEG {i:02d}] label={label:<20} sam2_score={float(sam_score):.4f} area={int(mask.sum())}")

    vis_image = draw_overlay(
        image=image,
        boxes=boxes,
        scores=det_scores,
        labels=labels,
        masks=masks,
    )

    vis_path = os.path.join(args.out_dir, f"{base_name}_open_vocab_seg.jpg")
    vis_image.save(vis_path)
    print(f"[SAVED] visualization: {vis_path}")

    if args.save_masks:
        mask_dir = os.path.join(args.out_dir, f"{base_name}_masks")
        mask_paths = save_masks(mask_dir, base_name, masks, labels)
        print(f"[SAVED] {len(mask_paths)} masks -> {mask_dir}")

    if args.save_json:
        json_path = os.path.join(args.out_dir, f"{base_name}_open_vocab_seg.json")
        save_json(json_path, boxes, det_scores, labels, sam_scores, masks)
        print(f"[SAVED] json: {json_path}")


if __name__ == "__main__":
    main()