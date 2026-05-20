import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import time
import albumentations as A

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.unet import UNet
from src.clrnet_seg import CLRNetSeg
from src.laneatt_seg import LaneATTSeg
from src.paths import CHECKPOINTS_DIR


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEIGHT, WIDTH = 360, 640

MODELS = {
    "U-Net": {
        "class": UNet,
        "checkpoint": CHECKPOINTS_DIR / "unet_best.pt",
    },
    "CLRNet-style": {
        "class": CLRNetSeg,
        "checkpoint": CHECKPOINTS_DIR / "clrnet_best.pt",
    },
    "LaneATT-style": {
        "class": LaneATTSeg,
        "checkpoint": CHECKPOINTS_DIR / "laneatt_best.pt",
    },
}


def load_model(model_name):
    info = MODELS[model_name]
    model = info["class"]().to(DEVICE)
    ckpt = torch.load(info["checkpoint"], map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


LOADED_MODELS = {name: load_model(name) for name in MODELS}


transform = A.Compose([A.Resize(HEIGHT, WIDTH)])


def predict(image_rgb, model_name, threshold):
    model = LOADED_MODELS[model_name]

    original = image_rgb.copy()
    h0, w0 = original.shape[:2]

    resized = transform(image=image_rgb)["image"]
    x = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        logits = model(x)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - t0) * 1000

    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)

    mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

    overlay = original.copy()
    overlay[mask > 0] = [0, 255, 0]
    result = cv2.addWeighted(overlay, 0.45, original, 0.55, 0)

    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    info = f"Model: {model_name}\nLatency: {latency_ms:.2f} ms\nFPS: {fps:.1f}"

    return result, info


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload road image"),
        gr.Dropdown(list(MODELS.keys()), value="U-Net", label="Model"),
        gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Threshold")
    ],
    outputs=[
        gr.Image(type="numpy", label="Lane Overlay"),
        gr.Textbox(label="Inference Info"),
    ],
    title="LaneDet: Real-time Lane Detection",
    description="Compare U-Net, CLRNet-style, and LaneATT-style lane segmentation models.",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)