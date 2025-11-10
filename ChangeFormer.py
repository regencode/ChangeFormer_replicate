import torch
import math
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timm.layers.drop import DropPath
import time


start = time.time()

#===========================================================

#           Helper functions
#   Shape conversions (N, C, W, H) -> (N, W*H, C)
#   (N, C, W, H) required by convs and output
#   (N, W*H, C) required by nn.Linear and Transformer blocks

#===========================================================

class Old_MHSA(nn.Module):

    def __init__(self, d_model=256, drop_path_rate=0.1, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model must be divisible with num_heads"
        d_head = d_model//num_heads
        self.d_head = d_head
        self.num_heads = num_heads
        self.drop_path = DropPath(drop_path_rate)

        self.q_proj = nn.Linear(d_model, d_head*num_heads)
        self.k_proj = nn.Linear(d_model, d_head*num_heads)
        self.v_proj = nn.Linear(d_model, d_head*num_heads)

        self.sdpa = SDPA(d_head)

    def forward(self, x: torch.Tensor):
        N, WH, C = x.shape
        d_head = self.d_head
        num_heads = self.num_heads
        d_model = d_head*num_heads

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(N, WH, num_heads, d_head).transpose(1, 2) #(N, num_heads, WH_reduced, d_head)
        K = K.view(N, WH, num_heads, d_head).transpose(1, 2)
        V = V.view(N, WH, num_heads, d_head).transpose(1, 2)

        x_attn = self.sdpa(Q, K, V)
        assert x_attn.shape == (N, num_heads, WH, d_head), x_attn.shape
        x_attn = x_attn.transpose(1, 2).flatten(start_dim=2)

        assert x_attn.shape == (N, WH, d_model), f"{x_attn.shape} != ({N}, {WH}, {d_model})"

        x = self.drop_path(x_attn) + x

        return x 

def test_old_mhsa():
    N, C, W, H = 1, 32, 64, 64
    x = torch.rand(N, W*H, C)
    mhsa = Old_MHSA(32)
    res = mhsa(x)
    assert res.shape == (N, W*H, C), res.shape
    print(f"Old MHSA assert passed | Elapsed time: {time.time() - start:.4f}s")

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

# helper to show a flattened sequence as image
def seq_to_image(seq, title="seq", S=None):
    # seq: (N, WH, C) or (N, WH, 1)
    with torch.no_grad():
        N, WH, C = seq.shape
        if S is None: S = int(WH**0.5)
        img = seq[0].permute(1,0).reshape(C, S, S)  # (C,H,W)
        # show first channel
        im = img[0].cpu().numpy()
        plt.figure(figsize=(4,4)); plt.title(title); plt.imshow((im - im.min())/(im.max()-im.min()+1e-8), cmap='viridis'); plt.axis('off'); plt.show()


#=============================================

#           Transformer Block
#   note: For transformer and nn.Linear 
#   permute C dimension to the back.

#=============================================

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim=512, kernel_size=7, stride=4, padding=0):
        super().__init__()
        self.conv_to_patch = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layernorm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.conv_to_patch(x)
        _, _, H, W = x.shape
        x = x.flatten(start_dim=-2).transpose(-2, -1)
        x = self.layernorm(x)
        return x

def test_patch_embed():
    N, C, W, H = 2, 3, 128, 128
    x = torch.rand(N, C, W, H)
    ope = OverlapPatchEmbed(3, embed_dim=32, kernel_size=7, stride=4, padding=3)
    x = ope(x)
    assert x.shape == (N, (W*H)//16, 32), x.shape
    print("Overlap patch embed assert pass")



class SDPA(nn.Module):
    '''
    no learnable parameters
    '''
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k ** -0.5

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        N, num_heads, HW, C = Q.shape
        logits = torch.matmul(Q, K.transpose(-2, -1))
        logits /= self.scale
        assert logits.shape == (N, num_heads, HW, HW), logits.shape

        attn = F.dropout(F.softmax(logits, dim=-1), p=0.1, training=self.training)
        return torch.matmul(attn, V)

def test_sdpa():
    N = 1
    C = 3
    H = 64
    W = 64
    q = torch.rand(N, 1, (H*W), C)
    k = torch.rand(N, 1, (H*W), C)
    v = torch.rand(N, 1, (H*W), C)

    sdpa = SDPA(C)
    res = sdpa(q, k, v)
    assert res.shape == (N, 1,H*W, C), res.shape
    print(f"SDPA assert passed | Elapsed time: {time.time() - start:.4f}s")

class SequenceReducer(nn.Module):
    def __init__(self, in_channels, reduce_ratio=4):
        super().__init__()
        self.reduce_ratio = reduce_ratio
        self.patch_conv = nn.Conv2d(in_channels, in_channels*reduce_ratio, kernel_size=reduce_ratio*2, stride=reduce_ratio)
        self.linear = nn.Conv2d(in_channels*reduce_ratio, in_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduce_ratio = self.reduce_ratio
        N, C, W, H = x.shape

        # patchify the image, such that local structure is preserved
        x = self.patch_conv(x)
        assert x.shape == (N, C*reduce_ratio, W//reduce_ratio, H//reduce_ratio), x.shape

        # project feature map such that C*reduce_ratio -> C
        x = self.linear(x)
        return x

def test_sr():
    sr = SequenceReducer(32, reduce_ratio=4)
    N, C, W, H = 1, 32, 64, 64
    x = torch.rand(N, C, W, H)
    res = sr(x)
    assert res.shape == (N, C, W//4, H//4), f"{res.shape} != (1, {C}, {W//4}, {H//4})"
    print(f"SequenceReducer assert passed | Elapsed time: {time.time() - start:.4f}s")

class MHSA(nn.Module):
    def __init__(self, d_model=256, drop_path_rate=0.1, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model must be divisible with num_heads"
        d_head = d_model//num_heads
        self.d_head = d_head
        self.num_heads = num_heads
        self.drop_path = DropPath(drop_path_rate)
        
        self.layernorm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_head*num_heads)
        self.k_proj = nn.Linear(d_model, d_head*num_heads)
        self.v_proj = nn.Linear(d_model, d_head*num_heads)

        self.sdpa = SDPA(d_head)

        self.attn_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        N, WH, C = x.shape
        d_head = self.d_head
        num_heads = self.num_heads
        d_model = d_head*num_heads

        #x = self.layernorm(x)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.reshape(N, WH, num_heads, d_head).transpose(1, 2) #(N, num_heads, WH_reduced, d_head)
        K = K.reshape(N, WH, num_heads, d_head).transpose(1, 2)
        V = V.reshape(N, WH, num_heads, d_head).transpose(1, 2)

        #x_attn = self.sdpa(Q, K, V)
        x_attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.1, is_causal=False)
        assert x_attn.shape == (N, num_heads, WH, d_head), x_attn.shape
        x_attn = x_attn.transpose(1, 2).flatten(start_dim=2)

        assert x_attn.shape == (N, WH, d_model), f"{x_attn.shape} != ({N}, {WH}, {d_model})"

        x = self.drop_path(self.attn_proj(x_attn)) + x
        return x 

def test_mhsa():
    N, C, W, H = 1, 32, 64, 64
    x = torch.rand(N, W*H, C)
    mhsa = MHSA(32)
    res = mhsa(x)
    assert res.shape == (N, W*H, C), res.shape
    print(f"MHSA assert passed | Elapsed time: {time.time() - start:.4f}s")


class PositionalEncoder(nn.Module):
    def __init__(self, in_channels, embed_dims=64, drop_path_rate=0.1):
        super().__init__()
        self.drop_path = DropPath(drop_path_rate)
        self.embed_dims = embed_dims
        self.mlp_in = nn.Linear(in_channels, embed_dims)
        self.conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, groups=embed_dims, padding=1, padding_mode="reflect") # depthwise conv
        self.gelu = nn.GELU()
        self.mlp_out = nn.Linear(embed_dims, in_channels)

        self.to_image = ConvertToForm("image")
        self.to_linear = ConvertToForm("linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, WH, C = x.shape
        x_in = self.mlp_in(x)
        x_in = self.to_image(x_in)
        x_in = self.gelu(self.conv(x_in))
        x_in = self.to_linear(x_in)
        return self.drop_path(self.mlp_out(x_in)) + x

def test_pos_enc():
    pe = PositionalEncoder(32, embed_dims=64)
    N, C, W, H = 1, 32, 64, 64
    x = torch.rand(N, W*H, C)
    res = pe(x)
    assert res.shape == x.shape, f"{res.shape} != {x.shape}"
    assert res.shape == (N, W*H, C), f"{res.shape}"
    print(f"PositionalEncoder assert passed | Elapsed time: {time.time() - start:.4f}s")


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, N=4, num_heads=4, pos_embed_dims=64, mhsa_drop_rate=0.1, posenc_drop_rate=0.1): 
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                MHSA(in_channels, 
                     drop_path_rate=mhsa_drop_rate, 
                     num_heads=num_heads
                ),
                PositionalEncoder(in_channels, embed_dims=pos_embed_dims, drop_path_rate=posenc_drop_rate)
            )
            for _ in range(N)
        ])
    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x


def test_transformer():
    N_blocks = 4
    transformer = TransformerBlock(32, N=N_blocks)
    N, C, W, H = 1, 32, 64, 64

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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="reflect")

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
    def __init__(self, in_channels, embed_dim=256):
        super().__init__()
        self.proj_pre = nn.Conv2d(in_channels, embed_dim, kernel_size=(1, 1))
        self.proj_post = nn.Conv2d(in_channels, embed_dim, kernel_size=(1, 1))

        self.conv1 = nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=(3, 3), padding=1, padding_mode="reflect")
        self.bn = nn.BatchNorm2d(num_features=embed_dim*2)
        self.conv2 = nn.Conv2d(embed_dim*2, embed_dim, kernel_size=(3, 3), padding=1, padding_mode="reflect")

    def forward(self, x_pre, x_post):
        x_pre = self.proj_pre(x_pre)
        x_post = self.proj_post(x_post)
        x = torch.cat([x_pre, x_post], dim=1) # concat along dim
        x = F.relu(self.bn(self.conv1(x)));
        return F.relu(self.conv2(x))

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
    def __init__(self, in_channels, embed_dims=256):
        super().__init__()
        self.embed_dims = embed_dims
        self.linear = nn.Conv2d(in_channels, embed_dims, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_dims)
        
    def forward(self, x, output_size):
        x = self.linear(x);
        x = F.interpolate(x, size=output_size, mode='nearest')
        return x

class MLPFusion(nn.Module):
    def __init__(self, embed_dims=64):
        super().__init__()
        self.linear = nn.Conv2d(embed_dims*4, embed_dims, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.bn(self.linear(x))

def test_mlp_upsampler_and_fusion():
    N, C, W, H = 1, 8, 128, 128
    F1 = torch.rand(N, C, W, H)
    F2 = torch.rand(N, C*2, W//2, H//2)
    F3 = torch.rand(N, C*4, W//4, H//4)
    F4 = torch.rand(N, C*8, W//8, H//8)

    mlp_up1 = MLPUpsampler(C, embed_dims=16)
    mlp_up2 = MLPUpsampler(C*2, embed_dims=16)
    mlp_up3 = MLPUpsampler(C*4, embed_dims=16)
    mlp_up4 = MLPUpsampler(C*8, embed_dims=16)

    res1 = mlp_up1(F1, output_size=(W, H))
    res2 = mlp_up2(F2, output_size=(W, H))
    res3 = mlp_up3(F3, output_size=(W, H))
    res4 = mlp_up4(F4, output_size=(W, H))

    assert res1.shape == (N, 16, W, H), res1.shape
    assert res2.shape == (N, 16, W, H), res2.shape
    assert res3.shape == (N, 16, W, H), res3.shape
    assert res4.shape == (N, 16, W, H), res4.shape

    print(f"MLP Upsampler assert passed | Elapsed time: {time.time() - start:.4f}s")

    fuse = MLPFusion(embed_dims=16)
    res = fuse(res1, res2, res3, res4)

    assert res.shape == (N, 16, W, H), res.shape
    print(f"MLP Fusion assert passed | Elapsed time: {time.time() - start:.4f}s")


class ConvUpsampleAndClassify(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dims=256):
        super().__init__()
        self.out_channels = out_channels

        # Diverge from original implementation, original implementation 
        # uses single Transposed Conv with kernel_size=4, stride=4 
        # to quadruple spatial dims

        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1)
        self.dense = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims, in_channels, kernel_size=3, padding=1)
        )
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1)
        self.conv_classify = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        N, C, W, H = x.shape
        x = self.conv1(x)
        assert x.shape == (N, C, W*2, H*2), x.shape
        x1 = self.dense(x)
        assert x1.shape == (N, C, W*2, H*2), x1.shape
        x = self.conv2(x + x1)
        assert x.shape == (N, C, W*4, H*4), x.shape
        class_logits = self.conv_classify(x)
        return class_logits

def test_conv_up_and_classify():
    N, C, W, H = 1, 16, 64, 64
    up_class = ConvUpsampleAndClassify(C, 3)
    x = torch.rand(N, C, W, H)
    res = up_class(x)
    assert res.shape == (N, 3, W*4, H*4), res.shape
    print(f"Conv Upsample and Classify assert passed | Elapsed time: {time.time() - start:.4f}s")

class ChangeFormerEncoder(nn.Module):
    def __init__(self, in_channels,
                 N=[3, 3, 4, 3], 
                 C=[64, 128, 256, 512], 
                 num_heads=[8, 8, 8, 8],
                 pos_embed_dims=[256, 256, 256, 256],
                 mhsa_drop_rates=[0.1, 0.1, 0.1, 0.1],
                 posenc_drop_rates=[0.1, 0.1, 0.1, 0.1]):
        super().__init__()
        self.b1 = nn.Sequential(
            OverlapPatchEmbed(in_channels, embed_dim=C[0], kernel_size=7, stride=4, padding=3),
            TransformerBlock(C[0], N=N[0], pos_embed_dims=pos_embed_dims[0],
                             mhsa_drop_rate=mhsa_drop_rates[0], posenc_drop_rate=posenc_drop_rates[0], 
                             num_heads=num_heads[0]),
            ConvertToForm("image")
        )
        self.b2 = nn.Sequential(
            OverlapPatchEmbed(C[0], embed_dim=C[1], kernel_size=3, stride=2, padding=1),
            TransformerBlock(C[1], N=N[1], pos_embed_dims=pos_embed_dims[1],
                             mhsa_drop_rate=mhsa_drop_rates[1], posenc_drop_rate=posenc_drop_rates[1],
                             num_heads=num_heads[1]),
            ConvertToForm("image")
        )
        self.b3 = nn.Sequential(
            OverlapPatchEmbed(C[1], embed_dim=C[2], kernel_size=3, stride=2, padding=1),
            TransformerBlock(C[2], N=N[2], pos_embed_dims=pos_embed_dims[2],
                             mhsa_drop_rate=mhsa_drop_rates[2], posenc_drop_rate=posenc_drop_rates[2],
                             num_heads=num_heads[2]),

            ConvertToForm("image")
        )
        self.b4 = nn.Sequential(
            OverlapPatchEmbed(C[2], embed_dim=C[3], kernel_size=3, stride=2, padding=1),
            TransformerBlock(C[3], N=N[3], pos_embed_dims=pos_embed_dims[3],
                             mhsa_drop_rate=mhsa_drop_rates[3], posenc_drop_rate=posenc_drop_rates[3],
                             num_heads=num_heads[3]),
            ConvertToForm("image")
        )

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        return x1, x2, x3, x4


class ChangeFormerDecoder(nn.Module):
    def __init__(self, num_classes,
                 C=[64, 128, 256, 512], 
                 embed_dims=256):
        super().__init__()
        self.diff1 = DifferenceModule(C[0], embed_dims)
        self.diff2 = DifferenceModule(C[1], embed_dims)
        self.diff3 = DifferenceModule(C[2], embed_dims)
        self.diff4 = DifferenceModule(C[3], embed_dims)

        self.mlp_up1 = MLPUpsampler(embed_dims, embed_dims=embed_dims)
        self.mlp_up2 = MLPUpsampler(embed_dims, embed_dims=embed_dims)
        self.mlp_up3 = MLPUpsampler(embed_dims, embed_dims=embed_dims)
        self.mlp_up4 = MLPUpsampler(embed_dims, embed_dims=embed_dims)

        self.mlp_fusion = MLPFusion(embed_dims)
        self.upsample_and_classify = ConvUpsampleAndClassify(embed_dims, num_classes)
    
    def forward(self, x1s, x2s, return_intermediate=False):
        x1_1, x1_2, x1_3, x1_4 = x1s
        x2_1, x2_2, x2_3, x2_4 = x2s

        F4 = self.diff4(x1_4, x2_4)
        F3 = self.diff3(x1_3, x2_3) + F.interpolate(F4, scale_factor=2, mode="bilinear")
        F2 = self.diff2(x1_2, x2_2) + F.interpolate(F3, scale_factor=2, mode="bilinear")
        F1 = self.diff1(x1_1, x2_1) + F.interpolate(F2, scale_factor=2, mode="bilinear")

        up1 = self.mlp_up1(F1, output_size=x1_1.shape[-2:])
        up2 = self.mlp_up2(F2, output_size=x1_1.shape[-2:])
        up3 = self.mlp_up3(F3, output_size=x1_1.shape[-2:])
        up4 = self.mlp_up4(F4, output_size=x1_1.shape[-2:])
        
        fused = self.mlp_fusion(up1, up2, up3, up4)
        out = self.upsample_and_classify(fused)

        if return_intermediate:
           return out, dict(F1=F1, F2=F2, F3=F3, F4=F4, up1=up1, up2=up2, up3=up3, up4=up4, fused=fused) 
        else:
            return out


class ChangeFormer(nn.Module):
    def __init__(self, in_channels, 
                 num_classes, 
                 N=[3, 3, 4, 3], 
                 C=[64, 128, 256, 512], 
                 embed_dims=256, 
                 num_heads=[8, 8, 8, 8],
                 pos_embed_dims=[256, 256, 256, 256],
                 mhsa_drop_rates=[0.1, 0.1, 0.1, 0.1],
                 posenc_drop_rates=[0.1, 0.1, 0.1, 0.1]
                 ):
        super().__init__()
        self.encoder = ChangeFormerEncoder(
            in_channels=in_channels,
            N=N,
            C=C,
            num_heads=num_heads,
            pos_embed_dims=pos_embed_dims,
            mhsa_drop_rates=mhsa_drop_rates,
            posenc_drop_rates=posenc_drop_rates
        )
        self.decoder = ChangeFormerDecoder(
            num_classes=num_classes,
            C=C,
            embed_dims=embed_dims
        )

    def forward(self, x1, x2):
        assert x1.shape == x2.shape, f"{x1.shape} != {x2.shape}"
        x1s = self.encoder(x1)
        x2s = self.encoder(x2)
        return self.decoder(x1s, x2s)


def test_changeformer():
    N, C, W, H = 1, 3, 64, 64
    num_classes = 2
    changeformer = ChangeFormer(C, num_classes, N=[2, 2, 2, 2])

    x1 = torch.rand(N, C, W, H)
    x2 = torch.rand(N, C, W, H)

    res = changeformer(x1, x2)
    assert res.shape == (N, num_classes, W, H), res.shape

    print(f"ChangeFormer assert passed | Elapsed time: {time.time() - start:.4f}s")


def visualize_feature_map(feat, title=None, num_channels=4):
    """
    feat: (N, C, H, W) tensor
    """
    with torch.no_grad():
        feat = feat[0]  # batch 0
        C = feat.shape[0]
        grid = min(num_channels, C)
        fig, axs = plt.subplots(1, grid, figsize=(3*grid, 3))
        if title:
            fig.suptitle(title)
        for i in range(grid):
            img = feat[i].detach().cpu()
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            axs[i].imshow(img, cmap='viridis')
            axs[i].axis('off')
        plt.show()

if __name__ == "__main__":
    test_conversion()
    test_patch_embed()
    test_sdpa()
    test_old_mhsa()
    test_mhsa()
    test_pos_enc()
    test_transformer()
    test_mlp_upsampler_and_fusion()
    test_conv_up_and_classify()
    test_changeformer()     
    print("All asserts pass.")
