import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)
class DimAlteration(nn.Module):
    def __init__(self, input_dim, output_dim, activation="ReLU"):
        super().__init__()
        if activation == "ReLU":
            self.dim_alteration = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            )
        elif activation == "Sigmoid":
            self.dim_alteration = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Sigmoid()
            )
        elif activation == "Linear":
            self.dim_alteration = nn.Linear(input_dim, output_dim)
        elif activation == "LayerNorm":                                                      
            self.dim_alteration = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:
            self.dim_alteration = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
    def forward(self, x):
        y = self.dim_alteration(x)
        return y
class GatedDimAlteration(nn.Module):
    def __init__(self, input_dim, output_dim, activation="ReLU"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if activation == "ReLU":
            self.dim_alteration = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            )
        elif activation == "Sigmoid":
            self.dim_alteration = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Sigmoid()
            )
        elif activation == "Linear":
            self.dim_alteration = nn.Linear(input_dim, output_dim)
        elif activation == "LayerNorm":                                                      
            self.dim_alteration = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:
            self.dim_alteration = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        self.need_gate = (input_dim != output_dim)
        if self.need_gate:
            self.input_proj = nn.Linear(input_dim, output_dim, bias=False)
            self.gate = nn.Linear(output_dim, output_dim)
            nn.init.zeros_(self.gate.bias)
    def forward(self, x):
        h = self.dim_alteration(x)
        if self.need_gate:
            x_proj = self.input_proj(x)                         
            z = torch.sigmoid(self.gate(h))                                
            h = (1 - z) * x_proj + z * h                   
        return h
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.dropout(attn_out)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)
        x = residual + ffn_out
        return x
class GatedEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.gate_attn = nn.Linear(d_model, d_model)
        self.gate_ffn  = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_attn.bias)
        nn.init.zeros_(self.gate_ffn.bias)
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.dropout(attn_out)
        z_a = torch.sigmoid(self.gate_attn(attn_out))
        x = (1 - z_a) * residual + z_a * attn_out
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)
        z_f = torch.sigmoid(self.gate_ffn(ffn_out))
        x = (1 - z_f) * residual + z_f * ffn_out
        return x
def auto_heads(d_model, min_heads=2, max_heads=16, min_dim_per_head=16):
    heads = d_model // min_dim_per_head
    heads = max(min_heads, min(heads, max_heads))
    while heads > 1 and d_model % heads != 0:
        heads -= 1
    return heads
class MultiHeadAttentionEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads=None, d_ff=None,
                 use_gated=False, dropout=0.3):
        super().__init__()
        if num_heads is None:
            num_heads = auto_heads(input_dim)
        if d_ff is None:
            d_ff = input_dim * 4
        BlockClass = GatedEncoderBlock if use_gated else EncoderBlock
        logger.info(
            f"  MHA: d_model={input_dim:>4d}, num_heads={num_heads:>2d}, "
            f"dim_per_head={input_dim // num_heads:>3d}, num_layers={num_layers}"
            f"{' [GATED]' if use_gated else ''}"
        )
        self.encoder_layers = nn.ModuleList([
            BlockClass(input_dim, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x
class DistanceMaskedEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, n_tokens, bandwidth, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(n_tokens)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        mask = dist <= bandwidth
        self.register_buffer('attn_mask', mask)
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=self.attn_mask)
        attn_out = self.dropout(attn_out)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)
        x = residual + ffn_out
        return x
class GatedDistanceMaskedEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, n_tokens, bandwidth, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(n_tokens)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        mask = dist <= bandwidth
        self.register_buffer('attn_mask', mask)
        self.gate_attn = nn.Linear(d_model, d_model)
        self.gate_ffn  = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_attn.bias)
        nn.init.zeros_(self.gate_ffn.bias)
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=self.attn_mask)
        attn_out = self.dropout(attn_out)
        z_a = torch.sigmoid(self.gate_attn(attn_out))
        x = (1 - z_a) * residual + z_a * attn_out
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)
        z_f = torch.sigmoid(self.gate_ffn(ffn_out))
        x = (1 - z_f) * residual + z_f * ffn_out
        return x
class DistanceMaskedMHAEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, n_tokens, bandwidth,
                 num_heads=None, d_ff=None, use_gated=False, dropout=0.3):
        super().__init__()
        if num_heads is None:
            num_heads = auto_heads(input_dim)
        if d_ff is None:
            d_ff = input_dim * 4
        BlockClass = GatedDistanceMaskedEncoderBlock if use_gated else DistanceMaskedEncoderBlock
        logger.info(
            f"  DistanceMaskedMHA: d_model={input_dim:>4d}, num_heads={num_heads:>2d}, "
            f"dim_per_head={input_dim // num_heads:>3d}, num_layers={num_layers}, "
            f"bandwidth={bandwidth}, n_tokens={n_tokens}"
            f"{' [GATED]' if use_gated else ''}"
        )
        self.encoder_layers = nn.ModuleList([
            BlockClass(input_dim, num_heads, d_ff, n_tokens, bandwidth, dropout=dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x
class TokenDeepEncoder(nn.Module):
    def __init__(self, num_features, stage_dims, bottleneck_dim,
                 activation="ReLU", to_bottleneck_activation=None,
                 feature_dim=None, use_gated=False, dropout=0.3):
        super().__init__()
        self.num_features = num_features
        if feature_dim is None:
            feature_dim = num_features
        self.feature_dim = feature_dim
        DimClass = GatedDimAlteration if use_gated else DimAlteration
        first_dim = stage_dims[0]
        self.input_proj = DimClass(feature_dim, first_dim, activation)
        self.pos_embed = nn.Embedding(num_features, first_dim)
        stage_modules = []
        n = len(stage_dims) - 1
        for i in range(n):
            stage_modules.append(DimClass(stage_dims[i], stage_dims[i + 1]))
            if i < n - 1:              
                stage_modules.append(
                    MultiHeadAttentionEncoder(stage_dims[i + 1], num_layers=2,
                                              use_gated=use_gated, dropout=dropout)
                )
        self.stages = nn.Sequential(*stage_modules)
        last_dim = stage_dims[-1]
        _bn_act = to_bottleneck_activation if to_bottleneck_activation is not None else activation
        self.to_bottleneck = DimAlteration(last_dim, bottleneck_dim, activation=_bn_act)
    def forward(self, x):
        tokens = self.input_proj(x)                                          
        positions = torch.arange(self.num_features, device=x.device)
        tokens = tokens + self.pos_embed(positions)                          
        tokens = tokens.unsqueeze(0)                                            
        tokens = self.stages(tokens)                                           
        tokens = tokens.squeeze(0)                                          
        bottleneck = self.to_bottleneck(tokens)                                   
        return bottleneck
class TokenReconstructor(nn.Module):
    def __init__(self, bottleneck_dim, num_features, stage_dims,
                 activation="Linear", feature_dim=None, use_gated=False,
                 hidden_activation=None, output_activation=None, dropout=0.3):
        super().__init__()
        self.num_features = num_features
        if feature_dim is None:
            feature_dim = num_features
        self.feature_dim = feature_dim
        _hidden = hidden_activation if hidden_activation is not None else activation
        _output = output_activation if output_activation is not None else activation
        DimClass = GatedDimAlteration if use_gated else DimAlteration
        first_dim = stage_dims[0]
        self.input_proj = DimClass(bottleneck_dim, first_dim, _hidden)
        self.pos_embed = nn.Embedding(num_features, first_dim)
        stage_modules = []
        n = len(stage_dims) - 1
        for i in range(n):
            stage_modules.append(DimClass(stage_dims[i], stage_dims[i + 1]))
            if i < n - 1:
                stage_modules.append(
                    MultiHeadAttentionEncoder(stage_dims[i + 1], num_layers=2,
                                              use_gated=use_gated, dropout=dropout)
                )
        self.stages = nn.Sequential(*stage_modules)
        last_dim = stage_dims[-1]
        self.to_features = nn.Linear(last_dim, feature_dim)
        if _output == "ReLU":
            self.output_activation = nn.ReLU()
        elif _output == "Softplus":
            self.output_activation = nn.Softplus()
        elif _output == "Sigmoid":
            self.output_activation = nn.Sigmoid()
        else:                
            self.output_activation = nn.Identity()
    def forward(self, z):
        tokens = self.input_proj(z)                                          
        positions = torch.arange(self.num_features, device=z.device)
        tokens = tokens + self.pos_embed(positions)                          
        tokens = tokens.unsqueeze(0)                                            
        tokens = self.stages(tokens)                                           
        tokens = tokens.squeeze(0)                                          
        output = self.to_features(tokens)                                      
        output = self.output_activation(output)
        return output
class TokenPredictor(nn.Module):
    def __init__(self, bottleneck_dim, num_output_tokens, stage_dims,
                 activation="Linear", feature_dim=None, use_gated=False,
                 hidden_activation=None, output_activation=None, dropout=0.3):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        if feature_dim is None:
            feature_dim = num_output_tokens
        self.feature_dim = feature_dim
        _hidden = hidden_activation if hidden_activation is not None else activation
        _output = output_activation if output_activation is not None else activation
        DimClass = GatedDimAlteration if use_gated else DimAlteration
        first_dim = stage_dims[0]
        self.input_proj = DimClass(bottleneck_dim, first_dim, _hidden)
        self.pos_embed = nn.Embedding(num_output_tokens, first_dim)
        stage_modules = []
        n = len(stage_dims) - 1
        for i in range(n):
            stage_modules.append(DimClass(stage_dims[i], stage_dims[i + 1]))
            if i < n - 1:
                stage_modules.append(
                    MultiHeadAttentionEncoder(stage_dims[i + 1], num_layers=2,
                                              use_gated=use_gated, dropout=dropout)
                )
        self.stages = nn.Sequential(*stage_modules)
        last_dim = stage_dims[-1]
        self.to_features = nn.Linear(last_dim, feature_dim)
        if _output == "ReLU":
            self.output_activation = nn.ReLU()
        elif _output == "Softplus":
            self.output_activation = nn.Softplus()
        elif _output == "Sigmoid":
            self.output_activation = nn.Sigmoid()
        else:                
            self.output_activation = nn.Identity()
    def forward(self, z):
        tokens = self.input_proj(z)                                          
        positions = torch.arange(self.num_output_tokens, device=z.device)
        tokens = tokens + self.pos_embed(positions)                          
        tokens = tokens.unsqueeze(0)                                            
        tokens = self.stages(tokens)                                           
        tokens = tokens.squeeze(0)                                          
        output = self.to_features(tokens)                                      
        output = self.output_activation(output)
        return output
class TokenTransformerBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_tokens, stage_dims,
                 activation="ReLU", use_gated=False, dropout=0.3):
        super().__init__()
        self.n_tokens = n_tokens
        first_dim = stage_dims[0]
        last_dim = stage_dims[-1]
        DimClass = GatedDimAlteration if use_gated else DimAlteration
        self.input_proj = DimClass(in_dim, first_dim, activation)
        self.pos_embed = nn.Embedding(n_tokens, first_dim)
        stage_modules = []
        n = len(stage_dims) - 1
        for i in range(n):
            stage_modules.append(DimClass(stage_dims[i], stage_dims[i + 1]))
            if i < n - 1:
                stage_modules.append(
                    MultiHeadAttentionEncoder(stage_dims[i + 1], num_layers=2,
                                              use_gated=use_gated, dropout=dropout))
        self.stages = nn.Sequential(*stage_modules)
        self.output_proj = DimAlteration(last_dim, out_dim, activation="Linear")
    def forward(self, x):
        x = self.input_proj(x)                                                      
        positions = torch.arange(self.n_tokens, device=x.device)
        x = x + self.pos_embed(positions)                                           
        x = x.unsqueeze(0)                                                             
        x = self.stages(x)                                                            
        x = x.squeeze(0)                                                           
        x = self.output_proj(x)                                                   
        return x
class Conv1DDownStage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 input_length, target_length, kernel_size=5):
        super().__init__()
        channel_list = [in_channels] + hidden_channels + [out_channels]
        n_layers = len(channel_list) - 1
        import math
        remaining = float(input_length) / float(target_length)
        strides = []
        for i in range(n_layers):
            s = max(2, int(round(remaining ** (1.0 / (n_layers - i)))))
            strides.append(s)
            remaining /= s
        layers = []
        for i in range(n_layers):
            layers.append(nn.Conv1d(
                channel_list[i], channel_list[i + 1],
                kernel_size, stride=strides[i],
                padding=kernel_size // 2
            ))
            layers.append(nn.BatchNorm1d(channel_list[i + 1]))
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)
    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
        return x
class Conv1DUpStage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 input_length, target_length, kernel_size=5):
        super().__init__()
        channel_list = [in_channels] + hidden_channels + [out_channels]
        n_layers = len(channel_list) - 1
        import math
        remaining = float(target_length) / float(input_length)
        strides = []
        for i in range(n_layers):
            s = max(2, int(round(remaining ** (1.0 / (n_layers - i)))))
            strides.append(s)
            remaining /= s
        layers = []
        for i in range(n_layers):
            layers.append(nn.ConvTranspose1d(
                channel_list[i], channel_list[i + 1],
                kernel_size, stride=strides[i],
                padding=kernel_size // 2,
                output_padding=strides[i] - kernel_size % 2
            ))
            if i < n_layers - 1:
                layers.append(nn.BatchNorm1d(channel_list[i + 1]))
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        self.target_length = target_length
    def forward(self, x):
        x = self.conv(x)
        if x.shape[-1] != self.target_length:
            x = nn.functional.interpolate(
                x, size=self.target_length,
                mode='linear', align_corners=False
            )
        return x
class Conv1DTransformerEncoder(nn.Module):
    def __init__(self, y_length, y_channels, x_p,
                 cnn_hidden, cnn_out,
                 transformer_stage_dims,
                 bottleneck, kernel_size=5, use_gated=False, dropout=0.3):
        super().__init__()
        self.x_p = x_p
        self.cnn = Conv1DDownStage(
            in_channels=y_channels,
            hidden_channels=cnn_hidden,
            out_channels=cnn_out,
            input_length=y_length,
            target_length=x_p,
            kernel_size=kernel_size)
        self.transformer = TokenTransformerBlock(
            in_dim=cnn_out, out_dim=bottleneck,
            n_tokens=x_p, stage_dims=transformer_stage_dims,
            activation="ReLU", use_gated=use_gated, dropout=dropout)
    def forward(self, y):
        x = self.cnn(y)                                          
        x = x.squeeze(0).permute(1, 0)                       
        u2 = self.transformer(x)                                
        return u2
class Conv1DTransformerDecoder(nn.Module):
    def __init__(self, bottleneck_dim, y_length, y_channels, x_p,
                 cnn_hidden, cnn_in,
                 transformer_stage_dims,
                 kernel_size=5, activation="Linear", use_gated=False,
                 hidden_activation=None, output_activation=None, dropout=0.3):
        super().__init__()
        _hidden = hidden_activation if hidden_activation is not None else activation
        _output = output_activation if output_activation is not None else activation
        self.transformer = TokenTransformerBlock(
            in_dim=bottleneck_dim, out_dim=cnn_in,
            n_tokens=x_p, stage_dims=transformer_stage_dims,
            activation=_hidden, use_gated=use_gated, dropout=dropout)
        self.cnn = Conv1DUpStage(
            in_channels=cnn_in,
            hidden_channels=cnn_hidden,
            out_channels=y_channels,
            input_length=x_p,
            target_length=y_length,
            kernel_size=kernel_size)
        if _output == "ReLU":
            self.output_activation = nn.ReLU()
        elif _output == "Softplus":
            self.output_activation = nn.Softplus()
        elif _output == "Sigmoid":
            self.output_activation = nn.Sigmoid()
        else:                
            self.output_activation = nn.Identity()
    def forward(self, z):
        x = self.transformer(z)                                    
        x = x.unsqueeze(0).permute(0, 2, 1)                          
        x = self.cnn(x)                                                        
        x = x.squeeze(0).permute(1, 0)                                     
        return self.output_activation(x)
