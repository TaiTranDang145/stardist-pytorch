import torch
import torch.nn as nn
import torch.nn.functional as F


# Double Conv
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# StarDist2D
class StarDist2D(nn.Module):
    def __init__(
        self,
        n_channels_in=1,
        n_rays=32,
        grid=(1, 1),
        unet_n_filter_base=32,
        unet_n_depth=3,
        net_conv_after_unet=128,
    ):
        super().__init__()
        self.n_rays = n_rays
        self.grid = grid

        # Grid pooling (tiền xử lý nếu grid > 1)
        self.input_pooling = nn.Identity()
        if any(g > 1 for g in grid):
            pooling_layers = []
            curr_grid = np.array([1, 1])
            target_grid = np.array(grid)
            in_c = n_channels_in
            while not np.all(curr_grid == target_grid):
                pool_size = (1 + (target_grid > curr_grid)).tolist()
                curr_grid *= pool_size
                # Thêm conv để feature alignment như bản gốc
                pooling_layers.append(nn.Conv2d(in_c, unet_n_filter_base, 3, padding=1))
                pooling_layers.append(nn.ReLU(inplace=True))
                pooling_layers.append(nn.MaxPool2d(pool_size))
                in_c = unet_n_filter_base
            self.input_pooling = nn.Sequential(*pooling_layers)
            in_ch = unet_n_filter_base
        else:
            in_ch = n_channels_in

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        out_ch = unet_n_filter_base

        for _ in range(unet_n_depth):
            self.down_blocks.append(DoubleConv(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch
            out_ch *= 2

        # Bottleneck
        self.bottleneck = DoubleConv(in_ch, out_ch)

        # Decoder
        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for _ in range(unet_n_depth):
            self.up_transpose.append(
                nn.ConvTranspose2d(out_ch, out_ch // 2, 2, stride=2)
            )
            self.up_blocks.append(
                DoubleConv(out_ch, out_ch // 2)
            )
            out_ch //= 2

        # Extra Conv
        if net_conv_after_unet > 0:
            self.features = nn.Sequential(
                nn.Conv2d(out_ch, net_conv_after_unet, 3, padding=1),
                nn.ReLU(inplace=True)
            )
            final_ch = net_conv_after_unet
        else:
            self.features = nn.Identity()
            final_ch = out_ch

        # Heads
        self.prob_head = nn.Sequential(
            nn.Conv2d(final_ch, 1, 1),
            nn.Sigmoid()
        )

        self.dist_head = nn.Conv2d(final_ch, n_rays, 1)

    def forward(self, x):
        # Grid pooling
        x = self.input_pooling(x)

        # Encoder
        skips = []

        for down, pool in zip(self.down_blocks, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up_trans, up_block, skip in zip(
            self.up_transpose,
            self.up_blocks,
            reversed(skips)
        ):
            x = up_trans(x)

            # pad nếu lệch size
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX//2, diffX - diffX//2,
                          diffY//2, diffY - diffY//2])

            x = torch.cat([skip, x], dim=1)
            x = up_block(x)

        # Feature
        feat = self.features(x)

        # Heads
        prob = self.prob_head(feat)
        dist = self.dist_head(feat)

        return prob, dist


# Test
if __name__ == "__main__":
    model = StarDist2D(n_channels_in=1, n_rays=32)
    dummy = torch.randn(2, 1, 256, 256)

    prob, dist = model(dummy)

    print("Prob shape:", prob.shape)
    print("Dist shape:", dist.shape)