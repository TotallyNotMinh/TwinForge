import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True, proj_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DinoVisionTransformer(nn.Module):
    def __init__(self, patch_size: int = 14, embed_dim: int = 384, depth: int = 12, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._pos_embed_cache = {}

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        cache_key = (w, h, x.device)
        if cache_key in self._pos_embed_cache:
            return self._pos_embed_cache[cache_key]

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        M = int(N ** 0.5)  # 37 for 518x518 default pos_embed
        patch_pos_embed = patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(h0, w0), mode="bicubic", align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        self._pos_embed_cache[cache_key] = pos_embed
        return pos_embed

    def forward_features(self, x: torch.Tensor, out_indices=(2, 5, 8, 11)):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        w0, h0 = x.shape[-1], x.shape[-2]
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, W, H)

        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in out_indices:
                features.append(x[:, 1:, :].permute(0, 2, 1).reshape(B, self.embed_dim, h0, w0))
        return features

class ResidualConvUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class RGBStem(nn.Module):
    def __init__(self, c_full: int = 16, c_half: int = 32, c_quarter: int = 64):
        super().__init__()
        self.stem_full = nn.Sequential(
            nn.Conv2d(3, c_full, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_full),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_full, c_full, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_full),
            nn.ReLU(inplace=True),
        )
        self.stem_half = nn.Sequential(
            nn.Conv2d(c_full, c_half, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_half, c_half, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.ReLU(inplace=True),
        )
        self.stem_quarter = nn.Sequential(
            nn.Conv2d(c_half, c_quarter, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_quarter),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_quarter, c_quarter, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_quarter),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        feat_full = self.stem_full(x)
        feat_half = self.stem_half(feat_full)
        feat_quarter = self.stem_quarter(feat_half)
        return feat_full, feat_half, feat_quarter

class DepthAnythingEncoder(nn.Module):
    """
    ViT-S encoder pre-trained on Depth Anything V2 (DINOv2 backbone).
    Extracts intermediate features and adapts them to multi-scale representations.
    """
    def __init__(
        self,
        checkpoint_path: str = "encoder_weights/depth_anything_v2_vits.pth",
        pretrained: bool = True,
        freeze: bool = True,
        freeze_early: bool = False,
        out_indices: tuple = (2, 5, 8, 11),
    ):
        super().__init__()
        self.vit = DinoVisionTransformer()

        if pretrained and checkpoint_path:
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.is_absolute():
                ckpt_path = Path(__file__).resolve().parent.parent / checkpoint_path
            if ckpt_path.exists():
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
                encoder_state = {k.replace("pretrained.", ""): v for k, v in ckpt.items() if k.startswith("pretrained.")}
                self.vit.load_state_dict(encoder_state)
                print(f"Loaded Depth Anything V2 ViT-S weights from {ckpt_path}")
            else:
                print(f"Warning: Checkpoint not found at {ckpt_path}, initializing randomly.")

        # RGB Stem
        self.rgb_stem = RGBStem(c_full=16, c_half=32)

        # Learned Reassembly Branches for taps [2, 5, 8, 11] (input dim = 384)
        out_c = 256
        self.resample_l2 = nn.ConvTranspose2d(384, out_c, kernel_size=4, stride=4)
        self.resample_l5 = nn.ConvTranspose2d(384, out_c, kernel_size=2, stride=2)
        self.resample_l8 = nn.Conv2d(384, out_c, kernel_size=1)
        self.resample_l11 = nn.Conv2d(384, out_c, kernel_size=3, stride=2, padding=1)

        # Post-resampling RCUs
        self.rcu_l2 = ResidualConvUnit(out_c)
        self.rcu_l5 = ResidualConvUnit(out_c)
        self.rcu_l8 = ResidualConvUnit(out_c)
        self.rcu_l11 = ResidualConvUnit(out_c)

        # Coarse-to-fine fusion RCUs
        self.fuse_rcu8 = ResidualConvUnit(out_c)
        self.fuse_rcu5 = ResidualConvUnit(out_c)
        self.fuse_rcu2 = ResidualConvUnit(out_c)

        # Channel projections matching MultiHeadDecoder expectations (256, 512, 1024, 2048)
        self.head_f2 = nn.Sequential(nn.Conv2d(out_c, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.head_f3 = nn.Sequential(nn.Conv2d(out_c, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.head_f4 = nn.Sequential(nn.Conv2d(out_c, 1024, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(inplace=True))
        self.head_f5 = nn.Sequential(nn.Conv2d(out_c, 2048, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU(inplace=True))        

        self.out_indices = out_indices
        self.freeze = freeze

        self.head_f1 = nn.Sequential(nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        if freeze:
            print("Encoder is completely frozen.")
            for p in self.vit.parameters():
                p.requires_grad = False
        elif freeze_early:
            print("Encoder is partially frozen.")
            for p in self.vit.patch_embed.parameters():
                p.requires_grad = False
            for blk in self.vit.blocks[:6]:
                for p in blk.parameters():
                    p.requires_grad = False


    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.vit.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        else:
            H, W = x.shape[-2:]

        # 1. RGB Stem features for edge preservation
        stem_full, stem_half, stem_quarter = self.rgb_stem(x)
            
        # Pad reflectively if dimensions are not divisible by 14
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # 2. Extract ViT tap features
        if self.freeze:
            with torch.no_grad():
                feats = self.vit.forward_features(x, out_indices=self.out_indices)
        else:
            feats = self.vit.forward_features(x, out_indices=self.out_indices)

        # 3. Learned Resampling + RCUs
        r2 = self.rcu_l2(self.resample_l2(feats[0]))
        r5 = self.rcu_l5(self.resample_l5(feats[1]))
        r8 = self.rcu_l8(self.resample_l8(feats[2]))
        r11 = self.rcu_l11(self.resample_l11(feats[3]))

        # 4. Top-Down Coarse-to-Fine Fusion
        p11 = r11
        p8 = self.fuse_rcu8(r8 + F.interpolate(p11, size=r8.shape[-2:], mode="bilinear", align_corners=False))
        p5 = self.fuse_rcu5(r5 + F.interpolate(p8, size=r5.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.fuse_rcu2(r2 + F.interpolate(p5, size=r2.shape[-2:], mode="bilinear", align_corners=False))

        # 5. Map to standard pyramid resolutions
        f5 = F.interpolate(self.head_f5(p11), size=(H // 32, W // 32), mode="bilinear", align_corners=False)
        f4 = F.interpolate(self.head_f4(p8), size=(H // 16, W // 16), mode="bilinear", align_corners=False)
        f3 = F.interpolate(self.head_f3(p5), size=(H // 8, W // 8), mode="bilinear", align_corners=False)
        f2 = F.interpolate(self.head_f2(p2), size=(H // 4, W // 4), mode="bilinear", align_corners=False)
        f1 = self.head_f1(stem_half)

        return {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "f5": f5,
            "stem_half": stem_half,
            "stem_full": stem_full,
            "stem_quarter": stem_quarter
        }

# Backward-compatible alias
ResNetEncoder = DepthAnythingEncoder
