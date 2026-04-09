import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms

# clase para cargar dataset

class LandfillDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, img_norm=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.img_norm = img_norm
        
        self.filenames = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.tif'))])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)

        if self.img_norm:
            image = self.img_norm(image)
            
        return image, mask

def get_dataloader(img_dir, mask_dir, batch_size, transform, img_norm, shuffle=True, num_workers=4):
    dataset = LandfillDataset(img_dir, mask_dir, transform=transform, img_norm=img_norm)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
