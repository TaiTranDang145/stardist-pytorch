import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Squeeze-and-Excitation Block (Channel Attention)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        z = self.squeeze(x).view(b, c)
        # Excitation: FC layers with ReLU and Sigmoid
        s = self.excitation(z).view(b, c, 1, 1)
        # Scale: multiply feature map with weights
        return x * s.expand_as(x)


# Attention Gate (Spatial Attention for Skip Connections)
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # g: gating signal from decoder, x: features from encoder
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Apply attention weights to input features
        return x * psi


# CBAM: Convolutional Block Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Channel attention
        x = x * self.channel_att(x)
        # Spatial attention
        x = x * self.spatial_att(x)
        return x


# Double Conv with SE Block
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_attention='se'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
        # Add attention mechanism
        if use_attention == 'se':
            self.attention = SEBlock(out_ch)
        elif use_attention == 'cbam':
            self.attention = CBAM(out_ch)
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        return x


# StarDist2D with Attention-Enhanced U-Net
class StarDist2D(nn.Module):
    def __init__(
        self,
        n_channels_in=1,
        n_rays=32,
        grid=(1, 1),
        unet_n_filter_base=32,
        unet_n_depth=3,
        net_conv_after_unet=128,
        n_harmonics=16,
        use_attention='se',  # 'se', 'cbam', or None
        use_attention_gate=True,
        use_boundary_head=True,
    ):
        super().__init__()
        self.n_rays = n_rays
        self.grid = grid
        self.n_harmonics = n_harmonics
        self.use_attention_gate = use_attention_gate
        self.use_boundary_head = use_boundary_head

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

        # Encoder with Attention
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        out_ch = unet_n_filter_base
        encoder_channels = []

        for _ in range(unet_n_depth):
            self.down_blocks.append(DoubleConv(in_ch, out_ch, use_attention=use_attention))
            self.pools.append(nn.MaxPool2d(2))
            encoder_channels.append(out_ch)
            in_ch = out_ch
            out_ch *= 2

        # Bottleneck with Attention
        self.bottleneck = DoubleConv(in_ch, out_ch, use_attention=use_attention)

        # Attention Gates (if enabled)
        self.attention_gates = nn.ModuleList()
        if use_attention_gate:
            # Build attention gates in reverse order (from deep to shallow)
            temp_ch = out_ch
            for i in range(unet_n_depth):
                F_g = temp_ch // 2  # gating signal channels (from decoder)
                F_l = encoder_channels[-(i+1)]  # encoder features channels
                F_int = F_l // 2  # intermediate channels
                self.attention_gates.append(AttentionGate(F_g, F_l, F_int))
                temp_ch = temp_ch // 2
        
        # Decoder with Attention
        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for _ in range(unet_n_depth):
            self.up_transpose.append(
                nn.ConvTranspose2d(out_ch, out_ch // 2, 2, stride=2)
            )
            self.up_blocks.append(
                DoubleConv(out_ch, out_ch // 2, use_attention=use_attention)
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

        # Output Heads
        # 1. Probability Head
        self.prob_head = nn.Sequential(
            nn.Conv2d(final_ch, 1, 1),
            nn.Sigmoid()
        )

        # 2. Distance Head
        self.dist_head = nn.Conv2d(final_ch, n_rays, 1)

        # 3. Boundary Head (NEW - as required in CHANGES.md)
        if use_boundary_head:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(final_ch, 1, 1),
                nn.Sigmoid()
            )
        
        # Optional Research Heads
        self.fourier_head = nn.Conv2d(final_ch, 2 * (n_harmonics + 1), 1)
        self.complexity_head = nn.Sequential(
            nn.Conv2d(final_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Grid pooling
        x = self.input_pooling(x)

        # Encoder with skip connections
        skips = []

        for down, pool in zip(self.down_blocks, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with Attention Gates
        for i, (up_trans, up_block, skip) in enumerate(zip(
            self.up_transpose,
            self.up_blocks,
            reversed(skips)
        )):
            x = up_trans(x)

            # Apply Attention Gate to skip connection if enabled
            if self.use_attention_gate:
                skip = self.attention_gates[i](g=x, x=skip)

            # Pad nếu lệch size
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX//2, diffX - diffX//2,
                          diffY//2, diffY - diffY//2])

            x = torch.cat([skip, x], dim=1)
            x = up_block(x)

        # Feature extraction
        feat = self.features(x)

        # Output Heads
        prob = self.prob_head(feat)
        dist = self.dist_head(feat)
        
        outputs = {
            'prob': prob,
            'dist': dist,
        }
        
        # Add boundary head if enabled
        if self.use_boundary_head:
            boundary = self.boundary_head(feat)
            outputs['boundary'] = boundary
        
        # Optional research heads
        fourier = self.fourier_head(feat)
        complexity = self.complexity_head(feat)
        outputs['fourier'] = fourier
        outputs['complexity'] = complexity

        return outputs


# Test
if __name__ == "__main__":
    print("Testing Attention-Enhanced StarDist2D...")
    
    # Test with SE Block and Attention Gates
    model = StarDist2D(
        n_channels_in=1, 
        n_rays=32, 
        n_harmonics=16,
        use_attention='se',
        use_attention_gate=True,
        use_boundary_head=True
    )
    dummy = torch.randn(2, 1, 256, 256)

    outputs = model(dummy)

    print("\n=== Model with SE + Attention Gates + Boundary Head ===")
    print("Prob shape:", outputs['prob'].shape)
    print("Dist shape:", outputs['dist'].shape)
    print("Boundary shape:", outputs['boundary'].shape)
    print("Fourier shape:", outputs['fourier'].shape)
    print("Complexity shape:", outputs['complexity'].shape)
    
    # Test with CBAM
    model_cbam = StarDist2D(
        n_channels_in=1, 
        n_rays=32, 
        n_harmonics=16,
        use_attention='cbam',
        use_attention_gate=True,
        use_boundary_head=True
    )
    
    outputs_cbam = model_cbam(dummy)
    print("\n=== Model with CBAM + Attention Gates + Boundary Head ===")
    print("Prob shape:", outputs_cbam['prob'].shape)
    print("Dist shape:", outputs_cbam['dist'].shape)
    print("Boundary shape:", outputs_cbam['boundary'].shape)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

