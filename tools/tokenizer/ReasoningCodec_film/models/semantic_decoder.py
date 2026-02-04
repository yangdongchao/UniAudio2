
import torch
import torch.nn as nn



class Conv1d1x1(nn.Conv1d):
    """1x1 Conv1d."""

    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv1d1x1, self).__init__(in_channels, out_channels, kernel_size=1, bias=bias)


class Conv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = -1,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).
        """
        x = self.conv(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            dilation=1,
            bias=False,
            nonlinear_activation="ELU",
            nonlinear_activation_params={},
    ):
        super().__init__()
        self.activation = getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
        )
        self.conv2 = Conv1d1x1(out_channels, out_channels, bias)

    def forward(self, x):
        y = self.conv1(self.activation(x))
        y = self.conv2(self.activation(y))
        return x + y


class ConvTranspose1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding=-1,
            output_padding=-1,
            groups=1,
            bias=True,
    ):
        super().__init__()
        if padding < 0:
            padding = (stride + 1) // 2
        if output_padding < 0:
            output_padding = 1 if stride % 2 else 0
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C', T').
        """
        x = self.deconv(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            dilations=(1, 1),
            unit_kernel_size=3,
            bias=True
    ):
        super().__init__()
        self.res_units = torch.nn.ModuleList()
        for dilation in dilations:
            self.res_units += [
                ResidualUnit(in_channels, in_channels,
                             kernel_size=unit_kernel_size,
                             dilation=dilation)
            ]
        self.num_res = len(self.res_units)

        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3 if stride == 1 else (2 * stride),  # special case: stride=1, do not use kernel=2
            stride=stride,
            bias=bias,
        )

    def forward(self, x):
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            input_channels: int,
            encode_channels: int,
            channel_ratios=(1, 1),
            strides=(1, 1),
            kernel_size=3,
            bias=True,
            block_dilations=(1, 1),
            unit_kernel_size=3
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)

        self.conv = Conv1d(
            in_channels=input_channels,
            out_channels=encode_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=False
        )
        self.conv_blocks = torch.nn.ModuleList()
        in_channels = encode_channels
        for idx, stride in enumerate(strides):
            out_channels = int(encode_channels * channel_ratios[idx])  # could be float
            self.conv_blocks += [
                EncoderBlock(in_channels, out_channels, stride,
                             dilations=block_dilations, unit_kernel_size=unit_kernel_size,
                             bias=bias)
            ]
            in_channels = out_channels
        self.num_blocks = len(self.conv_blocks)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        return x



class DecoderBlock(nn.Module):
    """ Decoder block (no up-sampling) """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            dilations=(1, 1),
            unit_kernel_size=3,
            bias=True
    ):
        super().__init__()

        if stride == 1:
            self.conv = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,  # fix kernel=3 when stride=1 for unchanged shape
                stride=stride,
                bias=bias,
            )
        else:
            self.conv = ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(2 * stride),
                stride=stride,
                bias=bias,
            )

        self.res_units = torch.nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            self.res_units += [
                ResidualUnit(out_channels, out_channels,
                             kernel_size=unit_kernel_size,
                             dilation=dilation)
            ]
        self.num_res = len(self.res_units)

    def forward(self, x):
        x = self.conv(x)
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            code_dim: int,
            output_channels: int,
            decode_channels: int,
            channel_ratios=(1, 1),
            strides=(1, 1),
            kernel_size=3,
            bias=True,
            block_dilations=(1, 1),
            unit_kernel_size=3,
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)

        self.conv1 = Conv1d(
            in_channels=code_dim,
            out_channels=int(decode_channels * channel_ratios[0]),
            kernel_size=kernel_size,
            stride=1,
            bias=False
        )

        self.conv_blocks = torch.nn.ModuleList()
        for idx, stride in enumerate(strides):
            in_channels = int(decode_channels * channel_ratios[idx])
            if idx < (len(channel_ratios) - 1):
                out_channels = int(decode_channels * channel_ratios[idx + 1])
            else:
                out_channels = decode_channels
            self.conv_blocks += [
                DecoderBlock(
                    in_channels, out_channels, stride,
                    dilations=block_dilations, unit_kernel_size=unit_kernel_size,
                    bias=bias
                )
            ]
        self.num_blocks = len(self.conv_blocks)

        self.conv2 = Conv1d(out_channels, output_channels, kernel_size, 1, bias=False)

    def forward(self, z):
        x = self.conv1(z)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        x = self.conv2(x)
        return x

class FiLM1d(nn.Module):
    """
    对 Conv1d 特征 (B, C, T) 做逐通道线性调制：
        out = x * (1 + γ) + β
    γ, β 由外部条件 cond ∈ ℝ^{cond_dim} 通过两层 MLP 预测
    """
    def __init__(self, in_channels: int, cond_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * in_channels)   # 输出 γ β
        )
        # 让网络初始表现为恒等映射
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x   : (B, C, T)
        cond: (B, cond_dim)
        """
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)          # (B, C) each
        # reshape 为 (B, C, 1) 便于 broadcast 到时间维
        gamma = gamma.unsqueeze(-1)
        beta  = beta.unsqueeze(-1)
        return x * (1 + gamma) + beta

class FiLM1dTemporal(nn.Module):
    """
    对 (B, D, T) 特征做逐通道 + 逐时间步调制：
        y(b, c, t) = x(b, c, t) * (1 + γ(b, c, t)) + β(b, c, t)
    γ, β 由条件张量 cond(b, T, cond_dim) → 1×1 Conv 预测
    """
    def __init__(self, channels: int, cond_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cond_dim, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden, 2 * channels, kernel_size=1)
        )
        # 让 FiLM 初始为恒等映射
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x   : (B, D, T)
        cond: (B, T, cond_dim)
        """
        cond = cond.permute(0, 2, 1)            # → (B, cond_dim, T)
        gamma_beta = self.net(cond)             # (B, 2D, T)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return x * (1 + gamma) + beta

class FiLMEncoder(Encoder):
    def __init__(self, phone_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)     # 调用上面给出的 Encoder
        self.film_in = FiLM1dTemporal(self.conv.out_channels, phone_dim)  # conv.out_channels == encode_channels

    def forward(self, x, phone_feat):
        """
        x             : (B, 1, T) 或其他输入
        phone_feat    : (B, phone_dim)
        """
        x = self.conv(x)                                         # (B, C, T)
        x = self.film_in(x, phone_feat)                                # 调制
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        return x
