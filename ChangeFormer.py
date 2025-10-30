import torch
import math
from torch import nn
import torch.nn.functional as F
import time

start = time.time()

#===========================================================

#           Helper functions
#   Shape conversions (N, C, W, H) -> (N, W*H, C)
#   (N, C, W, H) required by convs and output
#   (N, W*H, C) required by nn.Linear and Transformer blocks

#===========================================================

class ConvertToForm(nn.Module):
    def __init__(self, target_form: str):
        super().__init__()
        self.target_form = target_form
    def forward(self, x: torch.Tensor):
        if self.target_form == "image":
            '''
            convert to shape (N, C, W, H)
            '''
            N, WH, C = x.shape
            S = int(WH**0.5)
            x = x.permute(0, 2, 1)

            assert WH == S * S, f'''ConvertToForm("image") error: WH ({WH}) is not perfect square'''
            x = x.reshape((N, C, S, S))

            return x
        elif self.target_form == "linear":
            '''
            convert to shape (N, WH, C)
            '''
            N, C, W, H = x.shape
            x = x.flatten(start_dim=-2)
            x = x.permute(0, 2, 1)
            assert x.shape == (N, W*H, C), x.shape

            return x
        else:
            return -1

def test_conversion():
    N, C, H, W = 1, 32, 256, 256
    x = torch.rand(N, C, H, W)
    lin = ConvertToForm("linear")(x)
    img = ConvertToForm("image")(lin)
    assert torch.allclose(x, img), "Mapping mismatch"

    print("Conversion assert pass")

#=============================================

#           Transformer Block
#   note: For transformer and nn.Linear 
#   permute C dimension to the back.

#=============================================


class SDPA(nn.Module):
    '''
    no learnable parameters
    '''
    def __init__(self, d_head=64):
        super().__init__()
        self.scale = d_head ** 0.5

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        N, HW, C = Q.shape
        logits = torch.bmm(Q, K.transpose(1, 2))
        logits /= self.scale
        assert logits.shape == (N, HW, HW), logits.shape

        attn = F.softmax(logits, dim=-1)
        return torch.bmm(attn, V)

def test_sdpa():
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
    print(f"SDPA assert passed | Elapsed time: {time.time() - start:.4f}s")

class SequenceReducer(nn.Module):
    def __init__(self, in_channels, reduce_ratio=4):
        super().__init__()
        self.linear = nn.Linear(in_channels*reduce_ratio, in_channels)
        self.R = reduce_ratio
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, WH, C = x.shape

        h_new = (WH)//self.R
        assert WH == (WH//self.R) * self.R, f"{WH} not divisible by {self.R}"
        w_new = (C*self.R)
        x = self.linear(x.reshape(N, h_new, w_new))
        return x

def test_sr():
    sr = SequenceReducer(32, reduce_ratio=4)
    N, C, W, H = 1, 32, 64, 64
    x = torch.rand(N, W*H, C)
    res = sr(x)
    assert res.shape == (1, (H*W)/4, C), f"{res.shape} != (1, {(H*W)/4}, {C})"
    print(f"SequenceReducer assert passed | Elapsed time: {time.time() - start:.4f}s")

class MHSA(nn.Module):
    def __init__(self, in_channels, d_k=256, reduce_ratio=4):
        super().__init__()
        self.reduce_ratio = reduce_ratio
        self.d_k = d_k
        self.seq_reduce = SequenceReducer(in_channels, reduce_ratio)
        self.q_proj = nn.Linear(in_channels, d_k)
        self.k_proj = nn.Linear(in_channels, d_k)
        self.v_proj = nn.Linear(in_channels, d_k)
        self.sdpa = SDPA(d_k)
        self.out_proj = nn.Linear(d_k, in_channels*reduce_ratio)

    def forward(self, x: torch.Tensor):
        N, WH, C = x.shape
        reduce_ratio = self.reduce_ratio
        d_k = self.d_k

        x = self.seq_reduce(x)
        assert x.shape == (N, WH//reduce_ratio, C), x.shape

        x = self.sdpa(self.q_proj(x), self.k_proj(x), self.v_proj(x))
        assert x.shape == (N, WH//reduce_ratio, d_k), f"{x.shape} != ({N}, {WH//reduce_ratio}, {d_k})"
        
        x = self.out_proj(x) # learn some upsampling from d_k -> C*reduce_ratio
        x = x.reshape((N, WH, C)) # restore spatial dim

        assert x.shape == (N, WH, C), f"{x.shape} != ({N}, {WH}, {C})"
        return x 

def test_mhsa():
    N, C, W, H = 1, 32, 128, 128
    d_head = 16
    x = torch.rand(N, W*H, C)
    mhsa = MHSA(32, d_k=64, reduce_ratio=4)
    res = mhsa(x)
    assert res.shape == (N, W*H, 32), res.shape
    print(f"MHSA assert passed | Elapsed time: {time.time() - start:.4f}s")


class PositionalEncoder(nn.Module):
    def __init__(self, in_channels, embed_dims=64):
        super().__init__()
        self.embed_dims = embed_dims
        self.mlp_in = nn.Linear(in_channels, embed_dims)
        self.conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, groups=embed_dims, padding=1) # depthwise conv
        self.gelu = nn.GELU()
        self.mlp_out = nn.Linear(embed_dims, in_channels)

        self.to_image = ConvertToForm("image")
        self.to_linear = ConvertToForm("linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, WH, C = x.shape
        x_in = self.mlp_in(x)
        assert x_in.shape == (N, WH, self.embed_dims), x_in.shape

        assert WH == math.sqrt(WH) * math.sqrt(WH), f"{WH} != {math.sqrt(WH)} * {math.sqrt(WH)}"
        x_in = self.to_image(x_in)
        x_in = self.gelu(self.conv(x_in))
        x_in = self.to_linear(x_in)

        return self.mlp_out(x_in) + x

def test_pos_enc():
    pe = PositionalEncoder(32, embed_dims=64)
    N, C, W, H = 1, 32, 64, 64
    x = torch.rand(N, W*H, C)
    res = pe(x)
    assert res.shape == x.shape, f"{res.shape} != {x.shape}"
    assert res.shape == (N, W*H, C), f"{res.shape}"
    print(f"PositionalEncoder assert passed | Elapsed time: {time.time() - start:.4f}s")


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, N=4, reduce_ratio=16, d_head=64, pos_embed_dims=64): 
        super().__init__()
        self.reduce_ratio = reduce_ratio
        self.blocks = nn.ModuleList([
            nn.Sequential(
                MHSA(in_channels, d_k=d_head, reduce_ratio=reduce_ratio),
                PositionalEncoder(in_channels, embed_dims=pos_embed_dims)
            )
            for _ in range(N)
        ])
    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x


def test_transformer():
    reduce_ratio = 16
    N_blocks = 4
    transformer = TransformerBlock(32, N=N_blocks, reduce_ratio=reduce_ratio)
    N, C, W, H = 1, 32, 256, 256
    # Time Complexity per Reduced MHSA -> O((WH^2)/R)
    # Time Complexity per Transformer -> O(N(WH^2)/R)
    x = torch.rand(N, W*H, C)
    res = transformer(x)
    assert res.shape == (N, W*H, C), res.shape
    print(f"Transformer assert passed | Elapsed time: {time.time() - start:.4f}s")


#=================================================

#               Other blocks

#=================================================


class Downsampling(nn.Module):
    '''
    Halves spatial dimensions
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x);

def test_downsampling():
    N, C, W, H = 1, 32, 128, 128
    ds = Downsampling(32, 64)
    x = torch.rand(N, C, W, H)
    res = ds(x)
    assert res.shape == (N, 64, W/2, H/2), res.shape
    print(f"Downsampling assert passed | Elapsed time: {time.time() - start:.4f}s")


class DifferenceModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
    
    def forward(self, x_pre, x_post):
        x = torch.cat([x_pre, x_post], dim=1) # concat along dim
        x = self.conv(x);
        return self.bn(F.relu(x))

def test_difference_module():
    N, C, W, H = 1, 32, 128, 128
    diff = DifferenceModule(C*2, C)
    x1 = torch.rand(N, C, W, H)
    x2 = torch.rand(N, C, W, H)
    res = diff(x1, x2)
    assert res.shape == (N, C, W, H), res.shape
    print(f"DifferenceModule assert passed | Elapsed time: {time.time() - start:.4f}s")

#============================================================

#                   Decoder blocks

#============================================================

class MLPUpsampler(nn.Module):
    def __init__(self, in_channels, embed_dims=64, upsample_to_size=(128, 128)):
        super().__init__()
        self.embed_dims = embed_dims
        self.linear = nn.Linear(in_features=in_channels, out_features=embed_dims)
        self.output_size = upsample_to_size
        self.to_image = ConvertToForm("image")
        
    def forward(self, x):
        N, WH, C = x.shape
        x = self.linear(x);

        assert x.shape == (N, WH, self.embed_dims), x.shape
        assert WH == int(WH**0.5) * int(WH**0.5), f"{WH} != {int(WH**0.5)} * {int(WH**0.5)}"
        
        x = self.to_image(x)
        x = F.interpolate(x, size=self.output_size, mode='bilinear')
        return x

class MLPFusion(nn.Module):
    def __init__(self, embed_dims=64):
        super().__init__()
        self.linear = nn.Linear(in_features=embed_dims*4, out_features=embed_dims)
        self.to_linear = ConvertToForm("linear")

    def forward(self, x1, x2, x3, x4):
        N, C, W, H = x1.shape

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.to_linear(x)

        return self.linear(x)

def test_mlp_upsampler_and_fusion():
    N, C, W, H = 1, 8, 128, 128
    F1 = torch.rand(N, W*H, C)
    F2 = torch.rand(N, (W*H)//4, C*2)
    F3 = torch.rand(N, (W*H)//16, C*4)
    F4 = torch.rand(N, (W*H)//64, C*8)

    mlp_up1 = MLPUpsampler(C, embed_dims=16, upsample_to_size=(W, H))
    mlp_up2 = MLPUpsampler(C*2, embed_dims=16, upsample_to_size=(W, H))
    mlp_up3 = MLPUpsampler(C*4, embed_dims=16, upsample_to_size=(W, H))
    mlp_up4 = MLPUpsampler(C*8, embed_dims=16, upsample_to_size=(W, H))

    res1 = mlp_up1(F1)
    res2 = mlp_up2(F2)
    res3 = mlp_up3(F3)
    res4 = mlp_up4(F4)

    assert res1.shape == (N, 16, W, H), res1.shape
    assert res2.shape == (N, 16, W, H), res2.shape
    assert res3.shape == (N, 16, W, H), res3.shape
    assert res4.shape == (N, 16, W, H), res4.shape

    print(f"MLP Upsampler assert passed | Elapsed time: {time.time() - start:.4f}s")

    fuse = MLPFusion(embed_dims=16)
    res = fuse(res1, res2, res3, res4)

    assert res.shape == (N, W*H, 16), res.shape
    print(f"MLP Fusion assert passed | Elapsed time: {time.time() - start:.4f}s")


class ConvUpsampleAndClassify(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels

        # Diverge from original implementation, original implementation 
        # uses single Transposed Conv with kernel_size=4, stride=4 
        # to quadruple spatial dims

        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.to_linear = ConvertToForm("linear")
        self.to_image = ConvertToForm("image")

    def forward(self, x):
        N, C, W, H = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        assert x.shape == (N, C, W*4, H*4), x.shape

        x = self.to_linear(x)

        class_logits = self.linear(x)

        class_logits = self.to_image(class_logits)
        return class_logits

def test_conv_up_and_classify():
    N, C, W, H = 1, 16, 64, 64
    up_class = ConvUpsampleAndClassify(C, 3)
    x = torch.rand(N, C, W, H)
    res = up_class(x)
    assert res.shape == (N, 3, W*4, H*4), res.shape
    print(f"Conv Upsample and Classify assert passed | Elapsed time: {time.time() - start:.4f}s")

class ChangeFormer(nn.Module):
    def __init__(self, in_channels, num_classes, N=[4, 4, 4, 4], C=[16, 32, 64, 128], embed_dims=256, input_spatial_size=(256, 256), reduction_ratio=4, pos_embed_dims=[128, 128, 128, 128]):
        W, H = input_spatial_size
        super().__init__()
        self.b1 = nn.Sequential(
            Downsampling(in_channels, C[0], kernel_size=7, stride=4, padding=3),
            ConvertToForm("linear"),
            TransformerBlock(C[0], N=N[0], reduce_ratio=reduction_ratio, pos_embed_dims=pos_embed_dims[0]),
            ConvertToForm("image"),
        )
        self.b2 = nn.Sequential(
            Downsampling(C[0], C[1]),
            ConvertToForm("linear"),
            TransformerBlock(C[1], N=N[1], reduce_ratio=reduction_ratio, pos_embed_dims=pos_embed_dims[1]),
            ConvertToForm("image"),
        )
        self.b3 = nn.Sequential(
            Downsampling(C[1], C[2]),
            ConvertToForm("linear"),
            TransformerBlock(C[2], N=N[2], reduce_ratio=reduction_ratio, pos_embed_dims=pos_embed_dims[2]),
            ConvertToForm("image"),
        )
        self.b4 = nn.Sequential(
            Downsampling(C[2], C[3]),
            ConvertToForm("linear"),
            TransformerBlock(C[3], N=N[3], reduce_ratio=reduction_ratio, pos_embed_dims=pos_embed_dims[3]),
            ConvertToForm("image"),
        )
        self.diff1 = DifferenceModule(C[0]*2, C[0])
        self.diff2 = DifferenceModule(C[1]*2, C[1])
        self.diff3 = DifferenceModule(C[2]*2, C[2])
        self.diff4 = DifferenceModule(C[3]*2, C[3])

        self.mlp_up1 = MLPUpsampler(C[0], embed_dims=embed_dims, upsample_to_size=(W//4, H//4))
        self.mlp_up2 = MLPUpsampler(C[1], embed_dims=embed_dims, upsample_to_size=(W//4, H//4))
        self.mlp_up3 = MLPUpsampler(C[2], embed_dims=embed_dims, upsample_to_size=(W//4, H//4))
        self.mlp_up4 = MLPUpsampler(C[3], embed_dims=embed_dims, upsample_to_size=(W//4, H//4))

        self.to_image_form = ConvertToForm("image")
        self.to_linear_form = ConvertToForm("linear")

        self.mlp_fusion = MLPFusion(embed_dims)
        self.upsample_and_classify = ConvUpsampleAndClassify(embed_dims, num_classes)
    
    def forward(self, x1, x2):
        assert x1.shape == x2.shape, f"{x1.shape} != {x2.shape}"
        x1_1 = self.b1(x1)
        x1_2 = self.b2(x1_1)
        x1_3 = self.b3(x1_2)
        x1_4 = self.b4(x1_3)

        x2_1 = self.b1(x2)
        x2_2 = self.b2(x2_1)
        x2_3 = self.b3(x2_2)
        x2_4 = self.b4(x2_3)

        F1 = self.diff1(x1_1, x2_1)
        F2 = self.diff2(x1_2, x2_2)
        F3 = self.diff3(x1_3, x2_3)
        F4 = self.diff4(x1_4, x2_4)

        up1 = self.mlp_up1(self.to_linear_form(F1))
        up2 = self.mlp_up2(self.to_linear_form(F2))
        up3 = self.mlp_up3(self.to_linear_form(F3))
        up4 = self.mlp_up4(self.to_linear_form(F4))
        
        fused = self.mlp_fusion(up1, up2, up3, up4)
        return self.upsample_and_classify(self.to_image_form(fused))

def test_changeformer():
    N, C, W, H = 1, 3, 256, 256
    num_classes = 2
    changeformer = ChangeFormer(C, num_classes)

    x1 = torch.rand(N, C, W, H)
    x2 = torch.rand(N, C, W, H)

    res = changeformer(x1, x2)
    assert res.shape == (N, num_classes, W, H), res.shape

    print(f"ChangeFormer assert passed | Elapsed time: {time.time() - start:.4f}s")

if __name__ == "__main__":
    test_conversion()
    test_sdpa()
    test_changeformer()     
    print("All asserts pass.")



