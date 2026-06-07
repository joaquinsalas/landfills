import torch
import torch.nn.functional as F
import numpy as np
import supervision as sv
from PIL import Image
from pathlib import Path
import sys, os

sys.path.append('/home/emilio/Work/landfills/code/u-net_dinov3_sat/models')
sys.path.append('/home/emilio/Work/landfills/code/u-net_dinov3_sat/dataset')
sys.path.append('/home/emilio/Work/landfills/code/u-net_dinov3_sat/config')
sys.path.append(os.path.dirname(__file__))

import config as ini
from model_unet import UNet
from transform import get_transforms

INFERENCE_INPUT_DIR = '/home/emilio/Work/landfills/dataset/squared/val/landfills'
INFERENCE_THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, val_tf, img_norm = get_transforms(ini.INPUT_SIZE)

model = UNet().to(device)
model.load_state_dict(torch.load(f"/home/emilio/Work/landfills/models/dinov3_vit7b16_pretrain_sat493m-a6675841.pth", map_location=device))
model.eval()

def predict(img: Image.Image) -> np.ndarray:
    tensor = img_norm(val_tf(img, img)[0]).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        logits = model(tensor)
        
        logits_resized = F.interpolate(
            logits, size=(img.height, img.width), mode="bilinear", align_corners=False
        )
        mask = torch.sigmoid(logits_resized).squeeze().cpu().numpy()
        
    return mask > INFERENCE_THRESHOLD

def visualize(img_path: str):
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    
    mask = predict(img_pil)
    
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

    out_path = Path(ini.OUTPUT_DIR) / f"pred_{Path(img_path).stem}.png"
    Image.fromarray(annotated).save(out_path)
    print(f"saved: {out_path}")

if __name__ == "__main__":
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    images = [p for p in Path(INFERENCE_INPUT_DIR).iterdir() if p.suffix.lower() in exts]
    for img_path in images:
        visualize(str(img_path))
