import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class MLP(nn.Module):
    def __init__(self, in_dim, expansion):
        super().__init__()
        self.ln = nn.Sequential(
            nn.Linear(in_dim, in_dim * expansion),
            nn.ReLU(),
            nn.Linear(in_dim * expansion, in_dim)
        )

    def forward(self, x):
        return self.ln(x)


class PatchTransform(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, out_channels):
        super().__init__()
        # image_size = (image_size, image_size)
        # patch_size = (patch_size, patch_size)
        # patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        patches_resolution = [image_size // patch_size, image_size // patch_size]
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_emb = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size,
                                   stride=patch_size)

    def forward(self, x):
        # print(x.shape, self.in_channels, self.image_size)
        # x = self.patch_emb(x.view(x.shape[0], self.in_channels, self.image_size, self.image_size))  # (B, C, H, W)
        x = self.patch_emb(x.view(x.shape[0], self.image_size, self.image_size, self.in_channels)
                           .permute(0, 3, 1, 2))
        x = x.view(x.shape[0], self.out_channels, self.num_patches)  # (B, C, H * W // P^2)

        return x.permute(0, 2, 1)


class Head(nn.Module):
    def __init__(self, in_dim, head_size):
        super().__init__()
        self.key = nn.Linear(in_dim, head_size, bias=False)
        self.query = nn.Linear(in_dim, head_size, bias=False)
        self.value = nn.Linear(in_dim, head_size, bias=False)
        self.head_size = head_size

    def forward(self, x):
        k, q, v = self.key(x), self.value(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        out = wei @ v
        return out


class MultiHead(nn.Module):
    def __init__(self, in_dim, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(in_dim, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, in_dim)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)

        return out


class Block(nn.Module):
    def __init__(self, out_channels, group_size, num_heads, mlp_expansion):
        super().__init__()

        self.attn = MultiHead(out_channels, group_size, num_heads)
        self.mlp = MLP(out_channels, mlp_expansion)
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PyramidGroupBlock(nn.Module):
    def __init__(self, im_size, patch_size, in_channels, out_channels, group_size, num_heads, mlp_expansion,
                 num_blocks):
        super().__init__()
        self.im_size = im_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_size = group_size
        self.num_heads = num_heads
        self.mlp_expansion = mlp_expansion
        self.num_blocks = num_blocks

        self.patch_transform = PatchTransform(im_size, patch_size, in_channels, out_channels)
        self.blocks = nn.Sequential(*[Block(out_channels, group_size, num_heads, mlp_expansion)
                                      for _ in range(num_blocks)])

    def forward(self, x):
        patches = self.patch_transform(x)
        out = self.blocks(patches)
        return out.view(out.shape[0], self.im_size // self.patch_size, self.im_size // self.patch_size, self.out_channels)


class PyramidGroupTransformer(nn.Module):
    def __init__(self, im_size):
        super().__init__()
        self.channels = 64
        self.patch_size = 4
        self.stage_1 = PyramidGroupBlock(im_size=im_size, patch_size=self.patch_size, in_channels=1,
                                         out_channels=self.channels, group_size=64, num_heads=2, mlp_expansion=4,
                                         num_blocks=2)
        self.stage_2 = PyramidGroupBlock(im_size=im_size // 4, patch_size=2,
                                         in_channels=self.channels, out_channels=self.channels * 2, group_size=16,
                                         num_heads=4, mlp_expansion=4, num_blocks=2)
        self.stage_3 = PyramidGroupBlock(im_size=im_size // 8,
                                         patch_size=2,
                                         in_channels=self.stage_2.out_channels,
                                         out_channels=self.stage_2.out_channels * 2,
                                         group_size=1, num_heads=8, mlp_expansion=4, num_blocks=6)
        self.stage_4 = PyramidGroupBlock(im_size=im_size // 16,
                                         patch_size=2,
                                         in_channels=self.stage_3.out_channels,
                                         out_channels=self.stage_3.out_channels*2,
                                         group_size=1, num_heads=16, mlp_expansion=4, num_blocks=2)

    def forward(self, x):
        # print(x.shape)
        out1 = self.stage_1(x)# self.stage_1(x.view(x.shape[0], self.stage_1.im_size, self.stage_1.im_size, self.stage_1.in_channels))
        # print(out1.shape)
        out2 = self.stage_2(out1)
        # print(out2.shape)
        out3 = self.stage_3(out2)
        # print(out3.shape)
        out4 = self.stage_4(out3)
        # print(out4.shape)
        return out1, out2, out3, out4


class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, linear=False):
        super().__init__()
        self.linear = linear
        self.sr_ratio = sr_ratio
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim, eps=1e-5)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim, eps=1e-5)
            self.act = nn.GELU()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        q = self.q(x).reshape([B, N, self.num_heads, C // self.num_heads]).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).view(B, C, H, W)
                x_ = self.sr(x_).view(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape([B, -1, 2, self.num_heads, C // self.num_heads]).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape([B, -1, 2, self.num_heads, C // self.num_heads]).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).view(B, C, H, W)
            x_ = self.sr(self.pool(x_)).view(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape([B, -1, 2, self.num_heads, C // self.num_heads]).permute(2, 0, 3, 1, 4)

        k, v = kv
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = self.softmax(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, self.dim)
        out = self.proj(out)

        return out


class SpatialReductionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_expansion):
        super().__init__()

        self.attn = SpatialReductionAttention(dim, num_heads, 1)
        self.mlp = MLP(dim, mlp_expansion)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_expansion, num_blocks):
        super().__init__()

        self.sr_blocks = nn.ModuleList([SpatialReductionBlock(dim, num_heads, mlp_expansion) for _ in range(num_blocks)])

    def forward(self, img):
        for block in self.sr_blocks:
            B, C, H, W = img.shape
            img = F.interpolate(img, H * 2, mode='bilinear')
            img = block(img.permute(0, 2, 3, 1).view(B, -1, C))
            img = img.permute(0, 2, 1).view(B, C, H * 2, W * 2)
            print(img.shape)
        return img


class Decoder(nn.Module):
    def __init__(self, num_classes, dims=None):
        super().__init__()
        if dims is None:
            dims = [128, 256, 512]
        self.block1 = DecoderBlock(dims[2], num_heads=4, mlp_expansion=4, num_blocks=1)
        self.block2 = DecoderBlock(dims[2], num_heads=4, mlp_expansion=4, num_blocks=2)
        self.block3 = DecoderBlock(dims[2], num_heads=4, mlp_expansion=4, num_blocks=3)

        self.align1 = nn.Conv2d(in_channels=dims[0], out_channels=dims[2], kernel_size=1)
        self.align2 = nn.Conv2d(in_channels=dims[1], out_channels=dims[2], kernel_size=1)

        self.clf = nn.Conv2d(in_channels=dims[2], out_channels=num_classes, kernel_size=1)

    def forward(self, encoded):
        _, out1, out2, out3 = encoded
        out1 = out1.permute(0, 3, 1, 2)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = out3.permute(0, 3, 1, 2)
        print(out1.shape, out2.shape, out3.shape)
        x3 = out3
        x2 = self.align2(out2) + F.interpolate(x3, out2.shape[2], mode='bilinear')
        x1 = self.align1(out1) + F.interpolate(x2, out1.shape[2], mode='bilinear')
        out = self.block1(x1) + self.block2(x2) + self.block3(x3)
        logits = self.clf(out)
        logits = F.interpolate(logits, out.shape[2] * 4, mode='bilinear')
        return logits



def main():
    im_size = 224
    patch_size = 4
    in_channels = 3
    out_channels = 64
    group_size = 64
    num_heads = 2
    mlp_expansion = 4
    num_blocks = 1
    size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    x = torch.randn(1, im_size, im_size, in_channels)

    # m = PyramidGroupTransformer(im_size, patch_size, in_channels, out_channels, group_size,
    #                             num_heads, mlp_expansion, 1)
    m = PyramidGroupTransformer(im_size)
    summary(m)
    summary(Decoder(2))
    out = m(x)
    for out in m(x):
        print(out.shape)

    print(out[1].shape)
    print(F.interpolate(out[1].permute(0, 3, 1, 2), out[1].shape[2] * 2, mode='bilinear').shape)


if __name__ == '__main__':
    main()