import torch

weights_only=True
data = torch.load('./dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
print(data.keys)
