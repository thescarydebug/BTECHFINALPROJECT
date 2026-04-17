"""
URGMN model architecture — must match exactly what was trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

FEAT_COLS = [
    "CDR_GLOBAL", "CDR_SB", "MMSE_SCORE", "FAQ_TOTAL",
    "APOE_RISK",  "AGE",    "EDUCATION",  "COG_INDEX", "GENDER_BIN"
]
NUM_FEATS = len(FEAT_COLS)


class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // r), nn.ReLU(),
            nn.Linear(ch // r, ch), nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x).unsqueeze(-1).unsqueeze(-1)
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * self.sa(torch.cat([avg, mx], 1))


class ClinicalTransformer(nn.Module):
    def __init__(self, num_features=9, d_model=128,
                 nhead=4, num_layers=4, dropout=0.3):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, x):
        B = x.shape[0]
        tokens = torch.cat([
            p(x[:, i:i+1]).unsqueeze(1)
            for i, p in enumerate(self.projs)
        ], dim=1)
        cls    = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return self.out(self.transformer(tokens)[:, 0])


class URGMN(nn.Module):
    """
    Uncertainty and Reliability Guided Multimodal Network.
    Dual-stream: ResNet-18 + CBAM (MRI) + Transformer (clinical).
    Novel: Reliability Gate (N1), EDL head (N4).
    """
    def __init__(self, num_classes=3, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes

        # MRI stream — ResNet-18 with CBAM
        resnet = models.resnet18(weights=None)  # weights loaded from checkpoint
        self.stem   = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.cbam2  = CBAM(128)
        self.layer3 = resnet.layer3
        self.cbam3  = CBAM(256)
        self.layer4 = resnet.layer4
        self.cbam4  = CBAM(512)
        self.pool   = resnet.avgpool

        # Clinical stream — Transformer
        self.clin = ClinicalTransformer(
            num_features=NUM_FEATS, d_model=128,
            nhead=4, num_layers=4, dropout=dropout
        )

        # N1: Reliability Gate
        self.r_mri  = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(),
            nn.Linear(64, 1),   nn.Sigmoid()
        )
        self.r_clin = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),   nn.Sigmoid()
        )

        # Cross-attention fusion
        self.mri_proj  = nn.Linear(512, 256)
        self.clin_proj = nn.Linear(128, 256)
        self.cross_attn = nn.MultiheadAttention(
            256, num_heads=4, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)

        # N4: EDL head
        self.edl = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, clin):
        # MRI encoding
        x = self.stem(img)
        x = self.layer1(x)
        x = self.layer2(x); x = self.cbam2(x)
        x = self.layer3(x); x = self.cbam3(x)
        x = self.layer4(x); x = self.cbam4(x)
        mf = self.pool(x).flatten(1)          # [B, 512]

        # Clinical encoding
        cf = self.clin(clin)                   # [B, 128]

        # Reliability Gate (N1)
        r_mri  = self.r_mri(mf)               # [B, 1]
        r_clin = self.r_clin(cf)              # [B, 1]

        # Cross-attention
        mp = self.mri_proj(mf)                 # [B, 256]
        cp = self.clin_proj(cf)                # [B, 256]
        mp_n = self.norm1(mp.unsqueeze(1))
        cp_n = self.norm2(cp.unsqueeze(1))
        attn, _ = self.cross_attn(mp_n, cp_n, cp_n)
        attn = attn.squeeze(1)

        # Reliability-weighted fusion
        total_r = r_mri + r_clin + 1e-8
        fused   = (r_mri * mp + r_clin * attn) / total_r

        # EDL output (N4)
        ev  = F.softplus(self.edl(fused))
        alpha = ev + 1
        S     = alpha.sum(1, keepdim=True)
        probs = alpha / S
        uncertainty = float(self.num_classes) / S.squeeze()

        return probs, uncertainty, alpha