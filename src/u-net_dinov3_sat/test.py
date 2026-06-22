import torch
import torch.nn.functional as F
import numpy as np
import supervision as sv
from PIL import Image
from pathlib import Path

import config as ini
from dataset.transform import get_transforms
from models.model_unet import UNet

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def load_model(weights_path: Path, device: torch.device) -> UNet:
    model = UNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def predict(model: UNet, img: Image.Image, threshold: float, device: torch.device) -> np.ndarray:
    _, val_tf, img_norm = get_transforms(ini.INPUT_SIZE)
    tensor = img_norm(val_tf(img, img)[0]).unsqueeze(0).to(device)
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(tensor)

        logits_resized = F.interpolate(
            logits, size=(img.height, img.width), mode="bilinear", align_corners=False
        )
        mask = torch.sigmoid(logits_resized).squeeze().cpu().numpy()

    return mask > threshold


def visualize(model: UNet, img_path: Path, output_dir: Path, threshold: float, device: torch.device):
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)

    mask = predict(model, img_pil, threshold, device)

    if not mask.any():
        print(f"Sin detecciones (saltando anotación): {Path(img_path).name}")
        return

    y_indices, x_indices = np.where(mask)
    xyxy = np.array([[
        x_indices.min(),
        y_indices.min(),
        x_indices.max(),
        y_indices.max()
    ]])

    masks_array = np.expand_dims(mask, axis=0)
    detections = sv.Detections(xyxy=xyxy, mask=masks_array)

    mask_annotator = sv.MaskAnnotator(color=sv.Color.GREEN, opacity=0.4, color_lookup=sv.ColorLookup.INDEX)
    annotated = mask_annotator.annotate(scene=img_np.copy(), detections=detections)

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pred_{Path(img_path).stem}.png"
    Image.fromarray(annotated).save(out_path)
    print(f"saved: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ini.INFERENCE_WEIGHTS_PATH, device)
    images = [p for p in Path(ini.INFERENCE_INPUT_DIR).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]

    for img_path in images:
        visualize(model, img_path, ini.PREDICTIONS_DIR, ini.INFERENCE_THRESHOLD, device)


if __name__ == "__main__":
    main()
