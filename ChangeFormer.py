from functools import reduce
import torch
import math
import jaxtyping
from torch import nn
import torch.nn.functional as F

#=============================================

#           Transformer Block
#   note: For transformer permute C dimension
#   to the back.

#=============================================

class SDPA(nn.Module):
    '''
    no learnable parameters
    '''
    def __init__(self, d_head=64):
        super().__init__()
        self.heads = torch.tensor(d_head) # convert to tensor with 1 value for gpu processing

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        N, HW, C = Q.shape
        K = K.permute(0, 2, 1)
        logits = torch.bmm(Q, K)
        logits /= torch.sqrt(self.heads)

        assert logits.shape == (N, HW, HW), logits.shape
        softmaxed_logits = F.softmax(logits, dim=1)
        return softmaxed_logits @ V

N = 1
C = 3
H = 64
W = 64
q = torch.rand(N, (H*W), C)
k = torch.rand(N, (H*W), C)
v = torch.rand(N, (H*W), C)

sdpa = SDPA(d_head=64)
res = sdpa(q, k, v)
assert res.shape == (N, H*W, C), res
print("SDPA assert passed")

class MHSA(nn.Module):
    def __init__(self, in_channels, d_head=64):
        super().__init__()
        self.q_proj = nn.Linear(in_channels, d_head)
        self.k_proj = nn.Linear(in_channels, d_head)
        self.v_proj = nn.Linear(in_channels, d_head)
        self.sdpa = SDPA(d_head)
        self.out_proj = nn.Linear(d_head, in_channels)

    def forward(self, x):
        return self.out_proj(self.sdpa(self.q_proj(x), self.k_proj(x), self.v_proj(x)))

N, C, W, H = 1, 32, 128, 128
d_head = 16
x = torch.rand(N, W*H, C)
mhsa = MHSA(32, d_head=d_head)
res = mhsa(x)
assert res.shape == (N, W*H, 32), res.shape
print("MHSA assert passed")


class SequenceReducer(nn.Module):
    def __init__(self, in_channels, reduce_ratio=4):
        super().__init__()
        self.linear = nn.Linear(in_channels*reduce_ratio, in_channels)
        self.R = reduce_ratio
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, WH, C = x.shape

        h_new = (WH)//self.R
        w_new = (C*self.R)
        x = self.linear(x.reshape(N, h_new, w_new))
        return x

sr = SequenceReducer(32, reduce_ratio=4)
N, C, W, H = 1, 32, 64, 64
x = torch.rand(N, W*H, C)
res = sr(x)
assert res.shape == (1, (H*W)/4, C), f"{res.shape} != (1, {(H*W)/4}, {C})"
print("SequenceReducer assert passed")

class PositionalEncoder(nn.Module):
    def __init__(self, in_channels, embed_dims=64):
        super().__init__()
        self.embed_dims = embed_dims
        self.mlp_in = nn.Linear(in_channels, embed_dims)
        self.conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, groups=embed_dims, padding=1) # depthwise conv
        self.gelu = nn.GELU()
        self.mlp_out = nn.Linear(embed_dims, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, WH, C = x.shape
        x_in = self.mlp_in(x)
        assert x_in.shape == (N, WH, self.embed_dims), x_in.shape

        x_in = torch.permute(x_in, (0, 2, 1))
        assert WH == math.sqrt(WH) * math.sqrt(WH), f"{WH} != {math.sqrt(WH)} * {math.sqrt(WH)}"
        x_in = torch.reshape(x_in, (N, self.embed_dims, int(math.sqrt(WH)), int(math.sqrt(WH))))
        x_in = self.gelu(self.conv(x_in))
        x_in = torch.flatten(x_in, start_dim=2)
        x_in = torch.permute(x_in, (0, 2, 1))

        return self.mlp_out(x_in) + x

pe = PositionalEncoder(32, embed_dims=64)
N, C, W, H = 1, 32, 64, 64
x = torch.rand(N, W*H, C)
res = pe(x)
assert res.shape == x.shape, f"{res.shape} != {x.shape}"
assert res.shape == (N, W*H, C), f"{res.shape}"
print("PositionalEncoder assert passed")


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, N=4, reduce_ratio=4): 
        super().__init__()
        self.reduce_ratio = reduce_ratio
        self.blocks = nn.ModuleList([
                nn.Sequential(
                    SequenceReducer(in_channels, reduce_ratio=self.reduce_ratio),
                    MHSA(in_channels, d_head=64),
                    PositionalEncoder(in_channels, embed_dims=64)
                )
                for _ in range(N)
        ])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


reduce_ratio = 4
N_blocks = 4
transformer = TransformerBlock(32, N=N_blocks, reduce_ratio=reduce_ratio)
N, C, W, H = 1, 32, 256, 256
x = torch.rand(N, W*H, C)
res = transformer(x)
assert res.shape == (N, W*H/(reduce_ratio**N_blocks), C), res.shape
print("Transformer assert passed")


#=================================================

#               Other blocks

#=================================================


class Downsampling(nn.Module):
    '''
    Halves spatial dimensions
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x);


class DifferenceModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
    
    def forward(self, x_pre, x_post):
        x = torch.cat([x_pre, x_post], dim=1) # concat along dim
        x = self.conv(x);
        return self.bn(F.relu(x))

#============================================================

#                   Decoder blocks

#============================================================

class MLPUpsampler(nn.Module):
    def __init__(self, in_channels, embed_dims=64, upsample_to_size=(128, 128)):
        self.linear = nn.Linear(in_features=in_channels, out_features=embed_dims)
        self.output_size = upsample_to_size
        
    def forward(self, x):
        x = self.linear(x);
        x = F.interpolate(x, size=self.output_size, mode='bilinear')
        return x

class MLPFusion(nn.Module):
    def __init__(self, embed_dims=64):
        self.linear = nn.Linear(in_features=embed_dims*4, out_features=embed_dims)

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.linear(x)

class ConvUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=4)
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.linear(x)


class ChangeFormer(nn.Module):
    def __init__(self, in_channels, n_classes):
        self.b1 = nn.Sequential(
            Downsampling(in_channels, 32),
            TransformerBlock(32),
        )
        self.b2 = nn.Sequential(
            Downsampling(32, 64),
            TransformerBlock(64),
        )
        self.b3 = nn.Sequential(
            Downsampling(64, 128),
            TransformerBlock(128),
        )
        self.b4 = nn.Sequential(
            Downsampling(128, 256),
            TransformerBlock(256),
        )

        pass
        # to be implemented


print("All asserts pass.")
