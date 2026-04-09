import os
import gc
import cv2
import argparse
import numpy as np
import torch

from PIL import Image
from transformers import Sam2Model, Sam2Processor


class PointSegLocalApp:
    def __init__(
        self,
        image_path: str,
        model_name: str,
        device: str,
        precision: str,
        out_dir: str,
        max_display_size: int = 1400,
    ):
        self.image_path = image_path
        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.out_dir = out_dir
        self.max_display_size = max_display_size

        os.makedirs(self.out_dir, exist_ok=True)

        self.window_name = "Point-based Segmentation (SAM2)"

        self.image_pil = Image.open(self.image_path).convert("RGB")
        self.image_np_rgb = np.array(self.image_pil)
        self.image_np_bgr = cv2.cvtColor(self.image_np_rgb, cv2.COLOR_RGB2BGR)

        self.orig_h, self.orig_w = self.image_np_bgr.shape[:2]

        self.display_scale = self._compute_display_scale(
            self.orig_w, self.orig_h, self.max_display_size
        )
        self.display_w = int(round(self.orig_w * self.display_scale))
        self.display_h = int(round(self.orig_h * self.display_scale))

        self.base_display_bgr = cv2.resize(
            self.image_np_bgr,
            (self.display_w, self.display_h),
            interpolation=cv2.INTER_AREA if self.display_scale < 1.0 else cv2.INTER_LINEAR,
        )

        self.pos_points = []   # [[x, y], ...] on original image
        self.neg_points = []   # [[x, y], ...] on original image
        self.history = []      # [("pos", [x, y]), ("neg", [x, y]), ...]
        self.best_mask = None  # bool [H, W]
        self.best_score = None

        self.processor = None
        self.model = None

    def _compute_display_scale(self, w: int, h: int, max_size: int) -> float:
        long_side = max(w, h)
        if long_side <= max_size:
            return 1.0
        return max_size / long_side

    def _get_dtype(self):
        if self.precision == "fp32":
            return torch.float32
        if self.precision == "fp16":
            return torch.float16
        # auto
        if self.device.startswith("cuda"):
            # 为了本地老卡更稳，auto 默认也走 fp32
            return torch.float32
        return torch.float32

    def load_model(self):
        if self.processor is not None and self.model is not None:
            return

        dtype = self._get_dtype()
        print(f"[INFO] Loading SAM2 model: {self.model_name}")
        print(f"[INFO] Device: {self.device} | Precision: {dtype}")

        self.processor = Sam2Processor.from_pretrained(self.model_name)
        self.model = Sam2Model.from_pretrained(
            self.model_name,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def display_to_orig(self, x_disp: int, y_disp: int):
        x = int(round(x_disp / self.display_scale))
        y = int(round(y_disp / self.display_scale))
        x = max(0, min(self.orig_w - 1, x))
        y = max(0, min(self.orig_h - 1, y))
        return x, y

    def orig_to_display(self, x: int, y: int):
        x_disp = int(round(x * self.display_scale))
        y_disp = int(round(y * self.display_scale))
        x_disp = max(0, min(self.display_w - 1, x_disp))
        y_disp = max(0, min(self.display_h - 1, y_disp))
        return x_disp, y_disp

    def add_point(self, x_disp: int, y_disp: int, point_type: str):
        x, y = self.display_to_orig(x_disp, y_disp)

        if point_type == "pos":
            self.pos_points.append([x, y])
            self.history.append(("pos", [x, y]))
            print(f"[+ point] ({x}, {y})")
        else:
            self.neg_points.append([x, y])
            self.history.append(("neg", [x, y]))
            print(f"[- point] ({x}, {y})")

        self.best_mask = None
        self.best_score = None

    def undo_last_point(self):
        if len(self.history) == 0:
            print("[INFO] No points to undo.")
            return

        point_type, point = self.history.pop()
        if point_type == "pos" and point in self.pos_points:
            self.pos_points.remove(point)
        elif point_type == "neg" and point in self.neg_points:
            self.neg_points.remove(point)

        self.best_mask = None
        self.best_score = None
        print(f"[UNDO] Removed {point_type} point: {point}")

    def clear_points(self):
        self.pos_points = []
        self.neg_points = []
        self.history = []
        self.best_mask = None
        self.best_score = None
        print("[INFO] Cleared all points.")

    def build_inputs(self):
        all_points = self.pos_points + self.neg_points
        all_labels = [1] * len(self.pos_points) + [0] * len(self.neg_points)

        if len(all_points) == 0:
            raise ValueError("Please add at least one point first.")

        # Shape: [image_batch, object_batch, point_batch, 2]
        input_points = [[all_points]]
        input_labels = [[all_labels]]

        inputs = self.processor(
            images=self.image_pil,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        return inputs

    @torch.inference_mode()
    def run_segmentation(self):
        if len(self.pos_points) + len(self.neg_points) == 0:
            print("[WARN] No points provided.")
            return

        self.load_model()
        inputs = self.build_inputs()

        outputs = self.model(**inputs, multimask_output=True)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )[0]

        # 常见形状: [num_objects, num_masks, H, W]
        iou_scores = outputs.iou_scores[0, 0].detach().float().cpu().numpy()
        best_idx = int(np.argmax(iou_scores))
        self.best_score = float(iou_scores[best_idx])

        self.best_mask = masks[0, best_idx].numpy() > 0
        print(f"[SEGMENTED] best_score = {self.best_score:.4f}")

    def render_display_image(self):
        canvas = self.base_display_bgr.copy()

        # 先叠加 mask
        if self.best_mask is not None:
            mask_disp = cv2.resize(
                self.best_mask.astype(np.uint8),
                (self.display_w, self.display_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            overlay = np.zeros_like(canvas, dtype=np.uint8)
            overlay[:, :, 2] = 255  # red in BGR
            alpha = 0.35

            canvas[mask_disp] = cv2.addWeighted(
                canvas[mask_disp], 1 - alpha, overlay[mask_disp], alpha, 0
            )

        # 再画点
        for x, y in self.pos_points:
            xd, yd = self.orig_to_display(x, y)
            cv2.circle(canvas, (xd, yd), 7, (0, 255, 0), 2)
            cv2.line(canvas, (xd - 12, yd), (xd + 12, yd), (0, 255, 0), 2)
            cv2.line(canvas, (xd, yd - 12), (xd, yd + 12), (0, 255, 0), 2)

        for x, y in self.neg_points:
            xd, yd = self.orig_to_display(x, y)
            cv2.circle(canvas, (xd, yd), 7, (0, 0, 255), 2)
            cv2.line(canvas, (xd - 12, yd), (xd + 12, yd), (0, 0, 255), 2)
            cv2.line(canvas, (xd, yd - 12), (xd, yd + 12), (0, 0, 255), 2)

        # 信息条
        info1 = "L-click: positive | R-click: negative | Enter/Space: segment"
        info2 = "u: undo | c: clear | s: save | q/ESC: quit"
        score_txt = (
            f"score={self.best_score:.4f}"
            if self.best_score is not None
            else "score=N/A"
        )
        info3 = f"pos={len(self.pos_points)} | neg={len(self.neg_points)} | {score_txt}"

        cv2.putText(canvas, info1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(canvas, info2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(canvas, info3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

        return canvas

    def save_results(self):
        base = os.path.splitext(os.path.basename(self.image_path))[0]

        points_vis_path = os.path.join(self.out_dir, f"{base}_points_overlay.png")
        mask_path = os.path.join(self.out_dir, f"{base}_mask.png")

        vis = self.render_display_image()
        vis_save = cv2.resize(vis, (self.orig_w, self.orig_h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(points_vis_path, vis_save)

        if self.best_mask is not None:
            mask_img = (self.best_mask.astype(np.uint8) * 255)
            cv2.imwrite(mask_path, mask_img)
            print(f"[SAVED] mask: {mask_path}")
        else:
            print("[INFO] No mask yet. Only point overlay was saved.")

        print(f"[SAVED] overlay: {points_vis_path}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x, y, "pos")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.add_point(x, y, "neg")

    def run(self):
        print("=" * 80)
        print("Point-based Segmentation with SAM2")
        print("Left click  : add positive point")
        print("Right click : add negative point")
        print("Enter/Space : run segmentation")
        print("u           : undo last point")
        print("c           : clear all points")
        print("s           : save results")
        print("q / ESC     : quit")
        print("=" * 80)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_w, self.display_h)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            canvas = self.render_display_image()
            cv2.imshow(self.window_name, canvas)

            key = cv2.waitKey(20) & 0xFF

            if key in [13, 32]:  # Enter / Space
                try:
                    self.run_segmentation()
                except Exception as e:
                    print(f"[ERROR] {repr(e)}")

            elif key == ord("u"):
                self.undo_last_point()

            elif key == ord("c"):
                self.clear_points()

            elif key == ord("s"):
                self.save_results()

            elif key == ord("q") or key == 27:
                break

        cv2.destroyAllWindows()
        self.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/sam2.1-hiera-small")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="fp32", choices=["auto", "fp32", "fp16"])
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--max_display_size", type=int, default=1400)
    args = parser.parse_args()

    app = PointSegLocalApp(
        image_path=args.image_path,
        model_name=args.model_name,
        device=args.device,
        precision=args.precision,
        out_dir=args.out_dir,
        max_display_size=args.max_display_size,
    )
    app.run()


if __name__ == "__main__":
    main()