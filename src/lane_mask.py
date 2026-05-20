import cv2
import numpy as np


def lanes_to_mask(lanes, h_samples, height, width, line_thickness=5):
    mask = np.zeros((height, width), dtype=np.uint8)
    for lane in lanes:
        points = []
        for x, y in zip(lane, h_samples):
            if x < 0:
                continue
            points.append((int(x), int(y)))
        if len(points) >= 2:
            cv2.polylines(mask, [np.array(points, dtype=np.int32)], False, 1, line_thickness)
    return mask


def overlay_mask(image_bgr, mask, color=(0, 255, 0), alpha=0.5):
    overlay = image_bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
