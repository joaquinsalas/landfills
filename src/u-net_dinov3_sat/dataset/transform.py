import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F
from torchvision.transforms import InterpolationMode


def _rand_uniform(low, high):
    return torch.empty(1).uniform_(low, high).item()


def _has_foreground(mask):
    return F.to_image(mask).sum().item() > 0


def _sample_crop(mask, size, attempts=8):
    best = None
    needs_foreground = _has_foreground(mask)

    for _ in range(attempts):
        crop_scale = _rand_uniform(0.72, 1.0)
        crop_h = max(1, int(size[0] * crop_scale))
        crop_w = max(1, int(size[1] * crop_scale))
        top = int(torch.randint(0, size[0] - crop_h + 1, (1,)).item())
        left = int(torch.randint(0, size[1] - crop_w + 1, (1,)).item())
        best = top, left, crop_h, crop_w

        if not needs_foreground:
            break

        crop_mask = F.crop(mask, top, left, crop_h, crop_w)
        if _has_foreground(crop_mask):
            break

    return best


class SegmentationTransform:
    def __init__(self, input_size, augment=False):
        self.size = (input_size, input_size)
        self.augment = augment

    def __call__(self, image, mask):
        image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                image = F.horizontal_flip(image)
                mask = F.horizontal_flip(mask)
            if torch.rand(1).item() < 0.5:
                image = F.vertical_flip(image)
                mask = F.vertical_flip(mask)
            if torch.rand(1).item() < 0.75:
                angle = int(torch.randint(0, 4, (1,)).item()) * 90
                image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            if torch.rand(1).item() < 0.6:
                top, left, crop_h, crop_w = _sample_crop(mask, self.size)
                image = F.resized_crop(
                    image,
                    top,
                    left,
                    crop_h,
                    crop_w,
                    self.size,
                    interpolation=InterpolationMode.BILINEAR,
                )
                mask = F.resized_crop(
                    mask,
                    top,
                    left,
                    crop_h,
                    crop_w,
                    self.size,
                    interpolation=InterpolationMode.NEAREST,
                )
            if torch.rand(1).item() < 0.7:
                angle = _rand_uniform(-25.0, 25.0)
                translate = [
                    int(_rand_uniform(-0.08, 0.08) * self.size[1]),
                    int(_rand_uniform(-0.08, 0.08) * self.size[0]),
                ]
                scale = _rand_uniform(0.85, 1.15)
                shear = [_rand_uniform(-8.0, 8.0), _rand_uniform(-8.0, 8.0)]
                image = F.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
                mask = F.affine(
                    mask,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )
            if torch.rand(1).item() < 0.75:
                image = F.adjust_brightness(image, _rand_uniform(0.75, 1.25))
                image = F.adjust_contrast(image, _rand_uniform(0.75, 1.35))
                image = F.adjust_saturation(image, _rand_uniform(0.75, 1.25))
                image = F.adjust_hue(image, _rand_uniform(-0.04, 0.04))
            if torch.rand(1).item() < 0.25:
                kernel_size = int(torch.randint(3, 8, (1,)).item())
                if kernel_size % 2 == 0:
                    kernel_size += 1
                image = F.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])

        image = F.to_image(image)
        mask = F.to_image(mask)

        image = F.to_dtype(image, torch.float32, scale=True)
        mask = F.to_dtype(mask, torch.float32, scale=True)
        if self.augment and torch.rand(1).item() < 0.35:
            noise = torch.randn_like(image) * _rand_uniform(0.005, 0.025)
            image = torch.clamp(image + noise, 0.0, 1.0)
        mask = (mask > 0.5).float()

        return image, mask

def get_transforms(input_size):
    train_transform = SegmentationTransform(input_size, augment=True)
    val_test_transform = SegmentationTransform(input_size, augment=False)

    # normaliza 
    img_normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    return train_transform, val_test_transform, img_normalization
