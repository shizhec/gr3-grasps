"""Depth foreground baseline and optional language-conditioned detector.

The baseline does not read simulator body IDs, object poses or segmentation buffers.
It assumes a calibrated horizontal tabletop and a configured workspace.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class ObjectCloud:
    mask: np.ndarray
    points_camera: np.ndarray
    points_world: np.ndarray
    rgb: np.ndarray
    centroid_world: np.ndarray
    bbox_xyxy: list


def segment_tabletop(frame, config, box=None):
    points_camera = frame.points_camera()
    points_world = frame.points_world()
    lower, upper = np.array(config["workspace_min"]), np.array(config["workspace_max"])
    valid = np.isfinite(points_world).all(axis=-1) & (frame.depth_m > 0)
    valid &= ((points_world >= lower) & (points_world <= upper)).all(axis=-1)
    if box is not None:
        x1, y1, x2, y2 = np.asarray(box, dtype=int)
        roi = np.zeros(valid.shape, dtype=bool)
        x1, x2 = np.clip([x1, x2], 0, valid.shape[1])
        y1, y2 = np.clip([y1, y2], 0, valid.shape[0])
        roi[y1:y2, x1:x2] = True
        valid &= roi
    # Link tiny depth holes without dilating the points extracted for inference.
    linked = ndimage.binary_closing(valid, iterations=1)
    labels, count = ndimage.label(linked)
    clouds = []
    for label in range(1, count + 1):
        mask = (labels == label) & valid
        if np.count_nonzero(mask) < config["min_pixels"]:
            continue
        y, x = np.nonzero(mask)
        world = points_world[mask]
        clouds.append(ObjectCloud(mask, points_camera[mask], world, frame.rgb[mask],
                                  np.median(world, axis=0), [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]))
    return sorted(clouds, key=lambda c: tuple(c.centroid_world))


class GroundingDINO:
    """Optional open-vocabulary box detector. Used with depth foreground masking."""

    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device="cpu",
                 revision="a2bb814dd30d776dcf7e30523b00659f4f141c71"):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.torch, self.device = torch, device
        if device == "cpu":
            torch.set_num_threads(8)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, revision=revision).to(device).eval()

    def detect(self, rgb, text):
        from PIL import Image

        text = text.strip().rstrip(".") + "."
        inputs = self.processor(images=Image.fromarray(rgb), text=text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.3, text_threshold=0.25,
            target_sizes=[rgb.shape[:2]],
        )[0]
        if len(result["scores"]) == 0:
            raise RuntimeError(f"No object detected for {text!r}")
        best = int(result["scores"].argmax())
        return result["boxes"][best].detach().cpu().numpy()
