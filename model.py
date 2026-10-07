import torch 
import torch.nn as nn 
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
class mamba(nn.Module): 
    def __init__(self, dims, hidden_state= 16):
        super().__init__() 
        self.dims = dims 
        self.hidden_state= hidden_state
        self.A_log = nn.Parameter(torch.log(torch.rand(dims, hidden_state) + 1))
        self.B = nn.Linear(in_features= dims, out_features = hidden_state)  
        self.C= nn.Linear(in_features= dims, out_features = hidden_state)  
        self.D = nn.Parameter(torch.randn(dims))
        self.delta_lin  = nn.Linear(in_features= dims, out_features = dims)  
        self.delta_param = nn.Parameter(torch.randn(dims)) 
        self.tau = nn.Softplus()
    def forward(self, x):
        _, seq_len, _ = x.shape
        delta = self.tau(self.delta_param+ self.delta_lin(x))
        mat_A_delta = delta.unsqueeze(-1) * self.A_log                      
        A_bar = -torch.exp(mat_A_delta)
        B_bar = (1.0 / mat_A_delta) * (A_bar - 1) * (delta.unsqueeze(-1) * self.B(x).unsqueeze(2))
        C= self.C(x)
        h = torch.zeros(x.shape[0], self.dims, self.hidden_state,device=x.device)
        ys = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)   # (B,D,N)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)            # (B,D)     
            ys.append(y_t)
        y = torch.stack(ys, dim=1)    # (B,L,D)
        y= y+ self.D*x
        return y 
    
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(-1, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    c = windows.shape[-1]
    x = windows.view(-1, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, h, w, c)
    return x

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x
    
class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x
class ConvBlock(nn.Module):

    def __init__(self, 
                 dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x

class Block(nn.Module): 
    def __init__(self, 
                 dims,
                 num_heads, 
                 layer_scale, 
                 drop_path, 
                 mlp_ratio,
                 attn_drop,
                 drop,
                 norm_layer= nn.LayerNorm): 
        super().__init__()
        self.mamba = mamba(dims = dims)
        self.mlp1 = Mlp(in_features= dims,hidden_features= int(dims*mlp_ratio), drop = drop)
        self.attention = Attention(dim = dims, num_heads= num_heads, attn_drop= attn_drop)
        self.mlp2 =  Mlp(in_features= dims,hidden_features= int(dims*mlp_ratio), drop = drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dims))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dims))  if use_layer_scale else 1
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dims)
        self.norm1 = norm_layer(dims)
    def forward(self, x):
        x= x+ self.drop_path(self.gamma_1 * self.attention(self.norm2(x)))
        x= x+self.mlp1(x)
        x= x + self.drop_path(self.gamma_2 *self.mamba(self.norm1(x)))
        x= x+self.mlp2(x)
        return x 

class VisionLayer(nn.Module): 
    def __init__(self, 
                 dims, 
                 drop_path, 
                 depth,
                 downsample,
                 num_heads,
                 layer_scale, 
                 mlp_ratio, 
                 attn_drop, 
                 mlp_drop,
                 conv, 
                 window_size, 
                 layer_scale_conv= None, 
                 ): 
        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.Blocks = nn.ModuleList([ConvBlock(dim = dims, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else: 
            self.Blocks= nn.ModuleList([Block(dims = dims , num_heads= num_heads, layer_scale= layer_scale, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path , mlp_ratio = mlp_ratio, attn_drop= attn_drop, drop = mlp_drop )
                                        for i in range(depth)])
            self.transformer_block = True 
        self.window_size = window_size
        self.downsample = None if not downsample else Downsample(dim=dims)
    def forward(self, x): 
        _, _, H, W = x.shape
        
        if self.transformer_block:
            x= x.permute(0, 2, 3, 1)
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)
            b_, h,w ,c=  x.shape
            x= x.view(b_, h*w, c)
            for _, blk in enumerate(self.Blocks): 
                x= blk(x)
            x = window_reverse(x, self.window_size, Hp, Wp)
            x= x.permute(0, 3, 1, 2)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
             for _, blk in enumerate(self.Blocks): 
                x= blk(x)
        if self.downsample is not None: 
            return self.downsample(x)
        return x
            
class final_model(nn.Module): 
    def __init__(self, 
                 dims, 
                 depths, 
                 num_heads, 
                 mlp_ratio, 
                 window_size, 
                 num_classes, 
                 drop_path_rate, 
                 attn_drop_rate, 
                 drop_rate,
                 layer_scale=None,
                 layer_scale_conv=None,
                 
                 ): 
        super().__init__()
        num_features = int(dims * 2 ** (len(depths) - 1))
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)): 
             conv = True if (i == 0 ) else False
             level = VisionLayer(dims= int(dims * 2 ** i), 
                                 drop_path= dpr[sum(depths[:i]):sum(depths[:i + 1])], 
                                 depth = depths[i], 
                                 attn_drop= attn_drop_rate, 
                                 downsample= (i<1), 
                                 num_heads= num_heads[i], 
                                 layer_scale= layer_scale, 
                                 mlp_ratio= mlp_ratio, 
                                 mlp_drop=drop_rate, 
                                 conv= conv, 
                                 window_size= window_size[i])
             self.levels.append(level)
        self.patch_embed= PatchEmbed(dim= dims)
        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.softmax = nn.Softmax(dim = 1)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    def forward_features(self, x):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
# x= torch.randn(4,3, 32, 32)

# block = final_model(dims = 96,depths= [1,2], mlp_ratio= 0.6, window_size= [8, 4], num_classes= 10, drop_rate= 0.6, drop_path_rate= 0.4, attn_drop_rate= 0.6, num_heads= [6,6])
# block_= VisionLayer (dims = 96, drop_path= 0.4, depth = 3, downsample= False, num_heads= 6, layer_scale= None, mlp_ratio= 0.6, attn_drop= 0.4, mlp_drop= 0.4, conv = False, window_size= 8)
# y= block(x)

# print(y)
