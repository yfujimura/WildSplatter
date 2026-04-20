import torch
from torch import nn

from depth_anything_3.model.dinov2.layers import (  # noqa: F401
    Block,
    PatchEmbed,
    PositionGetter,
    RotaryPositionEmbedding2D,
    SwiGLUFFNFused,
)

class AppearanceEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.in_dim = cfg.in_dim
        self.out_dim = cfg.out_dim

        self.transformer_cfg = cfg.transformer
        self.head_cfg = cfg.head

        self.build_blocks()
        self.build_head()
        self.app_token = nn.Parameter(torch.randn(1, 1, self.in_dim))
        

    def build_head(self):
        num_layers = self.head_cfg.n_layers
        hidden_dim = self.head_cfg.hidden_dim

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers = []

        if num_layers == 1:
            # Single linear projection
            layers.append(nn.Linear(self.in_dim, self.out_dim, bias=True))
        else:
            # First
            layers.append(nn.Linear(self.in_dim, hidden_dim, bias=True))
            layers.append(nn.GELU())
            # Middle
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
                layers.append(nn.GELU())
            # Last
            layers.append(nn.Linear(hidden_dim, self.out_dim, bias=True))

        self.head = nn.Sequential(*layers)

    
    def build_blocks(
        self, 
        num_heads=12,
        mlp_ratio=2.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        norm_layer = nn.LayerNorm,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer=SwiGLUFFNFused,
        init_values=1.0,  # for layerscale: None or 0 => no layerscale
        qknorm_start=0,
        rope_start=0,
        rope_freq=100,
    ):
        embed_dim = self.in_dim
        depth = self.transformer_cfg.n_blocks
        self.patch_size = self.transformer_cfg.patch_size
        self.patch_start_idx = 1
        self.rope_start = rope_start

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None
        
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

    
    def _prepare_rope(self, B, H, W, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(
                B, H // self.patch_size, W // self.patch_size, device=device
            )
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B, self.patch_start_idx, 2).to(device).to(pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=1)
        return pos, pos_nodiff
        

    def forward(self, x, H, W):
        # x: (B, N, D)
        B = x.shape[0]
        
        app_token = self.app_token.expand(B, 1, -1)
        x = torch.cat([app_token, x], dim=1)
        
        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                pos = None
            elif i == self.rope_start:
                pos, _ = self._prepare_rope(B, H, W, x.device)
            x = blk(x, pos=pos, attn_mask=None)

        app_token = x[:,0] # (B, D)
        app_embed = self.head(app_token)
        return app_embed