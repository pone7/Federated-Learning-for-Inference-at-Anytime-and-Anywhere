from ast import arg
from turtle import pd
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _pair

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, recurrent_steps, heads, mlp_dim, dropout, depth=1):
        super().__init__()
        self.recurrent_steps = recurrent_steps
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for j in range(self.depth):
            for i in range(self.recurrent_steps):
                x = self.layers[j][0](x, mask = mask) # att
                x = self.layers[j][1](x) # ffn
        return x

class Accumulator(nn.Module):
    # def __init__(self, feature_length, num_classes, dim,
    #              recurrent_steps, depth, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
    def __init__(self, args, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.args = args
        self.feature_size = args.feature_size
        if 'base' in args.base_model:
            self.pos_embedding = nn.Parameter(torch.randn(1, 12+1, 768))
            self.dim = 768
        elif 'small' in args.base_model:
            self.pos_embedding = nn.Parameter(torch.randn(1, 24+1, 384))
            self.dim = 384
        elif 'tiny' in args.base_model:
            self.pos_embedding = nn.Parameter(torch.randn(1, 12+1, 192))
            self.dim = 192
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, 32+1, 1280))
            self.dim = 1280
        
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)

        # assert recurrent_steps * depth == 12, 'The architecture should match vanilla ViT'
        self.transformer = Transformer(self.dim, args.recurrent_steps, args.heads, args.mlp_dim, dropout, depth=args.depth)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, args.mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(args.mlp_dim, args.num_classes)
        )
        
    def batch_attn(self, x, is_training):
        if is_training:
            pre_x = x
            x = x.unsqueeze(1)
            x = self.batch_encoder(x)
            x = x.squeeze(1)
            return torch.cat([pre_x, x], dim=0)
        return x

    def forward(self, features, mask=None, cls=True):

        b, n, _ = features.shape
        residual_token = features[:, -1]
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, features), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)

        m = self.transformer(x, mask)

        if cls:
            return self.mlp_head(m[:,0]+residual_token)
        else:
            return m[:, -1]