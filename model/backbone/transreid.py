import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from typing import Optional, Tuple

# -------------------------- TransReID基础配置（对齐原论文与CSANet需求） --------------------------
TRANSREID_CONFIG = {
    "vit": {
        "img_size": (288, 144),  # VI-ReID常用输入尺寸（CSANet无修改）
        "patch_size": 16,         # ViT基础patch尺寸
        "in_chans": 3,            # 输入通道数（RGB/红外均按3通道处理）
        "embed_dim": 768,         # 嵌入维度（CSANet用此维度后续计算）
        "depth": 12,              # Transformer编码器层数
        "num_heads": 12,          # 注意力头数
        "mlp_ratio": 4.0,         # MLP隐藏层比例
        "dropout": 0.1,           #  dropout概率（防止过拟合）
        "drop_path_rate": 0.1     # DropPath概率
    },
    "bn": {
        "eps": 1e-5,              # 批归一化eps
        "momentum": 0.1           # 批归一化动量
    }
}

# -------------------------- 基础组件（原TransReID核心模块） --------------------------
class DropPath(nn.Module):
    """DropPath层（原TransReID用，替代Dropout增强泛化性）"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 按样本维度drop
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0或1
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP层（Transformer编码器 Feed-Forward Network）"""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """多头注意力层（原TransReID多头注意力实现）"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 需被num_heads {num_heads} 整除"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # 生成qkv：(B, N, 3*C) → (B, 3, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别取q, k, v

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力加权求和 + 投影
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer编码器块（Attention + MLP + 残差连接）"""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=drop, proj_drop=drop
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力块 + 残差
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        # MLP块 + 残差
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """图像→Patch嵌入层（将图像分割为Patch并映射到高维空间）"""
    def __init__(
        self,
        img_size: Tuple[int, int] = TRANSREID_CONFIG["vit"]["img_size"],
        patch_size: int = TRANSREID_CONFIG["vit"]["patch_size"],
        in_chans: int = TRANSREID_CONFIG["vit"]["in_chans"],
        embed_dim: int = TRANSREID_CONFIG["vit"]["embed_dim"]
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # 计算Patch数量：H//patch_size * W//patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # 卷积实现Patch分割与嵌入（替代全连接层，减少参数）
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        # 层归一化（原TransReID配置）
        self.norm = nn.LayerNorm(embed_dim, eps=TRANSREID_CONFIG["bn"]["eps"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像尺寸({H},{W})需匹配配置({self.img_size[0]},{self.img_size[1]})"
        
        # 图像→Patch嵌入：(B, C, H, W) → (B, embed_dim, H/patch, W/patch) → (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


# -------------------------- 双流Transformer特征提取器（CSANet专用） --------------------------
class TransReID_DualStream(nn.Module):
    """
    双流Transformer特征提取器（原TransReID基础上适配CSANet双模态需求）
    功能：分别处理可见光（Vis）与红外（IR）模态，输出模态共享特征（768维）
    """
    def __init__(
        self,
        feat_dim: int = TRANSREID_CONFIG["vit"]["embed_dim"],
        dropout: float = TRANSREID_CONFIG["vit"]["dropout"],
        num_heads: int = TRANSREID_CONFIG["vit"]["num_heads"],
        drop_path_rate: float = TRANSREID_CONFIG["vit"]["drop_path_rate"],
        pretrained: bool = True  # 是否加载TransReID预训练权重（CSANet论文默认）
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.pretrained = pretrained

        # -------------------------- 1. 双模态共享Patch嵌入层（Vis与IR输入格式一致） --------------------------
        self.patch_embed = PatchEmbed(
            img_size=TRANSREID_CONFIG["vit"]["img_size"],
            patch_size=TRANSREID_CONFIG["vit"]["patch_size"],
            in_chans=TRANSREID_CONFIG["vit"]["in_chans"],
            embed_dim=feat_dim
        )
        num_patches = self.patch_embed.num_patches

        # -------------------------- 2. 位置嵌入（Vis与IR共享位置信息） --------------------------
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, feat_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # -------------------------- 3. DropPath速率（按层递增，原TransReID策略） --------------------------
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, TRANSREID_CONFIG["vit"]["depth"])]

        # -------------------------- 4. Transformer编码器（双模态共享编码器，减少参数） --------------------------
        self.blocks = nn.ModuleList([
            Block(
                dim=feat_dim,
                num_heads=num_heads,
                mlp_ratio=TRANSREID_CONFIG["vit"]["mlp_ratio"],
                qkv_bias=True,  # 原TransReID启用qkv偏置
                drop=dropout,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm
            ) for i in range(TRANSREID_CONFIG["vit"]["depth"])
        ])
        self.norm = nn.LayerNorm(feat_dim)

        # -------------------------- 5. 模态专属BN层（适配双模态分布差异，CSANet新增） --------------------------
        self.bn_vis = nn.BatchNorm1d(feat_dim, eps=TRANSREID_CONFIG["bn"]["eps"], momentum=TRANSREID_CONFIG["bn"]["momentum"])
        self.bn_ir = nn.BatchNorm1d(feat_dim, eps=TRANSREID_CONFIG["bn"]["eps"], momentum=TRANSREID_CONFIG["bn"]["momentum"])
        # BN层权重初始化（原TransReID策略）
        init.normal_(self.bn_vis.weight.data, 1.0, 0.01)
        init.normal_(self.bn_ir.weight.data, 1.0, 0.01)
        init.zeros_(self.bn_vis.bias.data)
        init.zeros_(self.bn_ir.bias.data)

        # -------------------------- 6. 初始化权重（含预训练权重加载） --------------------------
        self.apply(self._init_weights)
        if self.pretrained:
            self._load_pretrained_weights()  # 加载TransReID预训练权重（需自行准备权重文件）

    def _init_weights(self, m: nn.Module) -> None:
        """权重初始化（原TransReID初始化策略）"""
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.zeros_(m.bias)
            init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                init.zeros_(m.bias)

    def _load_pretrained_weights(self) -> None:
        """加载TransReID预训练权重（示例逻辑，需替换为实际权重路径）"""
        try:
            pretrained_path = "./pretrained/transreid_vit_base_patch16.pth"  # 权重文件路径
            state_dict = torch.load(pretrained_path, map_location="cpu")
            # 匹配权重（排除模态专属BN层，因预训练无此部分）
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and "bn_vis" not in k and "bn_ir" not in k}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print("成功加载TransReID预训练权重")
        except FileNotFoundError:
            print("未找到TransReID预训练权重，将使用随机初始化权重")

    def forward_features(self, x: torch.Tensor, modal: str) -> torch.Tensor:
        """提取单模态特征（Vis/IR共用逻辑，仅最后BN层区分模态）"""
        # 1. Patch嵌入 + 位置嵌入
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed  # 加位置嵌入
        x = self.pos_drop(x)    # 位置Dropout

        # 2. Transformer编码器特征提取
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # 最终层归一化

        # 3. 全局平均池化（Patch特征→图像特征）
        x = x.mean(dim=1)  # (B, embed_dim)

        # 4. 模态专属BN层（适配双模态分布差异，CSANet核心适配点）
        if modal == "vis":
            x = self.bn_vis(x)
        elif modal == "ir":
            x = self.bn_ir(x)
        else:
            raise ValueError(f"模态错误：仅支持'modal=\"vis\"'（可见光）或'modal=\"ir\"'（红外）")

        return x

    def forward(self, x: torch.Tensor, modal: str) -> torch.Tensor:
        """
        前向传播（CSANet调用接口）
        Args:
            x: 输入图像张量（shape=[B, 3, H, W]，H=288, W=144）
            modal: 模态标识（"vis"=可见光，"ir"=红外）
        Returns:
            feat: 模态共享特征（shape=[B, 768]）
        """
        feat = self.forward_features(x, modal)
        return feat