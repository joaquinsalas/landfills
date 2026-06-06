import torch
import torch.nn as nn
import torch.nn.functional as F
import config as ini
from backbone import build_backbone
from DoubleConv import DoubleConv
from decoder import Up

class ProjectionBridge(nn.Module):
    def __init__(self, in_ch, out_ch, up_factor=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.up_factor = up_factor

    def forward(self, x):
        x = self.conv(x)
        if self.up_factor > 1:
            x = F.interpolate(x, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.backbone = build_backbone(ini.BACKBONE_NAME, getattr(ini, 'BACKBONE_PATH', None))
        
        self.D = getattr(self.backbone, 'embed_dim', 1024)
        depth = len(self.backbone.blocks) if hasattr(self.backbone, 'blocks') else 24
        
        self._hook_idxs = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        self._hook_features = {}
        for i in self._hook_idxs:
            self.backbone.blocks[i].register_forward_hook(
                lambda m, inp, out, i=i: self._hook_features.update({i: out})
            )

        # Reconstrucción de la pirámide espacial escalando dinámicamente las salidas del ViT
        self.projs = nn.ModuleList([
            ProjectionBridge(self.D, 1024, up_factor=1),
            ProjectionBridge(self.D, 512,  up_factor=2),
            ProjectionBridge(self.D, 256,  up_factor=4),
            ProjectionBridge(self.D, 128,  up_factor=8),
        ])
        
        # Extracción de la conexión "shallow" directo de la imagen para garantizar la máxima resolución
        self.proj_shallow = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512,  256)
        self.up3 = Up(256,  128)
        self.up4 = Up(128,   64)
        
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        self._hook_features.clear()
        self.backbone(x)

        P = self.backbone.patch_embed.patch_size[0] if hasattr(self.backbone, 'patch_embed') else 16
        h, w = H // P, W // P

        def to_map(feat):
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            extra_tokens = feat.shape[1] - (h * w)
            return feat[:, extra_tokens:].transpose(1, 2).reshape(x.shape[0], self.D, h, w)

        skips = [to_map(self._hook_features[i]) for i in self._hook_idxs]
        
        projs = [p(s) for p, s in zip(self.projs, reversed(skips))]
        shallow = self.proj_shallow(x)

        out = self.up1(projs[0], projs[1])
        out = self.up2(out, projs[2])
        out = self.up3(out, projs[3])
        out = self.up4(out, shallow)
        
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, (H, W), mode='bilinear', align_corners=False)
            
        return self.outc(out)
