import torch
import torch.nn as nn
import torch.nn.functional as F
import config as ini
from backbone import build_backbone
from DoubleConv import DoubleConv
from decoder import Up

class ProjectionBridge(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.use_backbone = ini.BACKBONE_TYPE != 'none'

        if self.use_backbone:
            D = ini.EMBED_DIM
            self.backbone = build_backbone(ini.BACKBONE_NAME, ini.BACKBONE_PATH, ini.BACKBONE_FREEZE)

            # ViT-L: 24 bloques -> hooks en [6,12,18,23]
            depth = len(self.backbone.blocks)
            idxs = [depth // 4, depth // 2, 3 * depth // 4, depth - 1]
            self._hook_features = {}
            for i in idxs:
                self.backbone.blocks[i].register_forward_hook(
                    lambda m, inp, out, i=i: self._hook_features.update({i: out})
                )
            self._hook_idxs = idxs

            self.projs = nn.ModuleList([ProjectionBridge(D, ch) for ch in [1024, 512, 256, 128]])
            self.proj_shallow = ProjectionBridge(D, 64)
            self.up1 = Up(1024, 512)
            self.up2 = Up(512,  256)
            self.up3 = Up(256,  128)
            self.up4 = Up(128,   64)
        else:
            from encoder import Down
            self.inc   = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128);  self.down2 = Down(128, 256)
            self.down3 = Down(256, 512); self.down4 = Down(512, 1024)
            self.up1   = Up(1024, 512);  self.up2   = Up(512, 256)
            self.up3   = Up(256, 128);   self.up4   = Up(128, 64)

        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        if not self.use_backbone:
            x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2)
            x4=self.down3(x3); x5=self.down4(x4)
            return self.outc(self.up4(self.up3(self.up2(self.up1(x5,x4),x3),x2),x1))

        self._hook_features = {}
        self.backbone(x)  # populate hooks

        h = w = ini.INPUT_SIZE // ini.PATCH_SIZE
        B, D = x.shape[0], ini.EMBED_DIM

        def to_map(feat):
            return feat[:, 1:].transpose(1,2).reshape(B, D, h, w)  # quitar cls

        skips = [to_map(self._hook_features[i]) for i in self._hook_idxs]
        projs = [p(s) for p, s in zip(self.projs, reversed(skips))]  # profundo->shallow
        shallow = self.proj_shallow(skips[0])

        x = self.up1(projs[0], projs[1])
        x = self.up2(x, projs[2])
        x = self.up3(x, projs[3])
        x = self.up4(x, shallow)
        return self.outc(F.interpolate(x, (H, W), mode='bilinear', align_corners=False))
