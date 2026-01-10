import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels//16, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels//16, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        se_weight = self.se(out)
        out = out * se_weight
        
        out += residual
        return torch.relu(out)

class CNNModel(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        tower_layers = [
            nn.Conv2d(143, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        ]
        num_res_blocks = 8 
        for _ in range(num_res_blocks):
            tower_layers.append(ResidualBlock(256, 256))

        tower_layers += [
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 64),
            ResidualBlock(64, 32),
            nn.Flatten()
        ]
        self._tower = nn.Sequential(*tower_layers)

        # 保持与原版相同的输出维度
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 9, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(512, 235)
        )
        
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 9, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # 更精细的参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        hidden = self._tower(obs)
        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value = self._value_branch(hidden)
        return masked_logits, value

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)  # 缩小eps，提升稳定性
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src_norm = self.norm1(src)
        attn_output, _ = self.self_attn(src_norm, src_norm, src_norm)  # 自注意力
        src = src + self.dropout1(attn_output)
        
        src_norm = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ff_output)
        return src

class TransformerModel(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        # 1. 轻量化CNN嵌入（适配小尺寸特征图）
        self.conv_embed = nn.Sequential(
            # nn.Conv2d(143, 256, 3, 1, 1), 
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            nn.Conv2d(143, 128, 1, 1, 0),  # 1×1卷积先降维，保留局部特征
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 1, 1, 0),  # 二次1×1升维，避免空间信息丢失
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 9))  # 强制保持输出尺寸，兼容后续序列转换
        )
        
        # 2. Transformer配置
        self.d_model = 256
        self.seq_len = 36  # 4*9
        self.nhead = 4
        self.dim_feedforward = self.d_model * 4  # 常规设置：d_model×4
        
        # 位置编码：改进初始化 + 可学习
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, self.d_model) * 0.02)  # 小幅值初始化
        
        # 3. Transformer编码器
        self.transformer_encoder = nn.Sequential(
            TransformerBlock(self.d_model, self.nhead, self.dim_feedforward, dropout=dropout),
            TransformerBlock(self.d_model, self.nhead, self.dim_feedforward, dropout=dropout),
            TransformerBlock(self.d_model, self.nhead, self.dim_feedforward, dropout=dropout),
            TransformerBlock(self.d_model, self.nhead, self.dim_feedforward, dropout=dropout)
        )

        # 4. 输出头（对齐CNNModel，增加归一化和正则化）
        self._logits = nn.Sequential(
            nn.Linear(self.d_model * self.seq_len, 512),
            nn.BatchNorm1d(512),  # 增加BN
            nn.ReLU(),
            nn.Dropout(dropout),  # 增加Dropout
            nn.Linear(512, 235)
        )
        
        self._value_branch = nn.Sequential(
            nn.Linear(self.d_model * self.seq_len, 512),
            nn.BatchNorm1d(512),  # 增加BN
            nn.GELU(),
            nn.Dropout(dropout),  # 增加Dropout
            nn.Linear(512, 256),
            nn.Tanh(),  # 对齐CNNModel的值域约束
            nn.Linear(256, 1)
        )

        # 5. 参数初始化（补充Transformer特有层）
        self._init_weights()

    def _init_weights(self):
        """统一初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        # 设备对齐
        device = next(self.parameters()).device
        obs = input_dict["observation"].float().to(device)
        mask = input_dict["action_mask"].float().to(device)
        
        # 维度断言（调试用）
        assert obs.shape[1:] == (143, 4, 9), f"Obs shape error: {obs.shape}, expected (B,143,4,9)"
        assert mask.shape[1] == 235, f"Mask shape error: {mask.shape}, expected (B,235)"
        
        # 1. CNN嵌入
        x = self.conv_embed(obs)  # [B, 256, 4, 9]
        
        # 2. 重塑为序列 [B, C, H, W] -> [B, H*W, C]
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, -1, c)  # 更直观的维度变换
        
        # 3. 位置编码
        x = x + self.pos_encoder  # 广播机制：[1,36,256] + [B,36,256]
        
        # 4. Transformer编码
        x = self.transformer_encoder(x)  # [B, 36, 256]
        
        # 5. 展平
        x = x.reshape(b, -1)  # [B, 36*256]
        
        # 6. 输出头（改进mask处理，避免NaN）
        logits = self._logits(x)
        # 安全的mask处理：替换log(0)为-1e9，避免inf
        inf_mask = torch.where(mask == 0, torch.tensor(-1e9, device=device), torch.tensor(0.0, device=device))
        masked_logits = logits + inf_mask
        
        # 7. Value分支
        value = self._value_branch(x)
        
        return masked_logits, value
    
class TransformerMultiHeadModel(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        # 1. 轻量化CNN嵌入（适配小尺寸特征图）
        self.conv_embed = nn.Sequential(
            nn.Conv2d(143, 256, 3, 1, 1),  # 1×1卷积替代3×3，减少计算量
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        
        # 2. Transformer配置
        self.d_model = 256
        self.seq_len = 36  # 4*9
        self.nhead = 4
        self.dim_feedforward = self.d_model * 4  # 常规设置：d_model×4
        
        # 位置编码：改进初始化 + 可学习
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, self.d_model) * 0.02)  # 小幅值初始化
        self.row_embed = nn.Parameter(torch.randn(1, 4, 1, self.d_model) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, 1, 9, self.d_model) * 0.02)

        # 3. Transformer编码器
        self.transformer_encoder = nn.Sequential(
            TransformerBlock(self.d_model, self.nhead, self.dim_feedforward, dropout=dropout),
            TransformerBlock(self.d_model, self.nhead, self.dim_feedforward, dropout=dropout),
            TransformerBlock(self.d_model, self.nhead, self.dim_feedforward, dropout=dropout)
        )

        # 4. 输出头（修改为multi heads）
        # Head 1: Play (Discard) - 负责出牌，对应 Action 索引 2-35 (共34张牌)
        self.head_play = nn.Sequential(
            nn.Linear(self.d_model*self.seq_len, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 34)
        )
        
        # Head 2: Interaction - 负责交互 (Pass, Hu, Chi, Peng, Gang...)
        # 对应 Action 索引 0-1 (Pass, Hu) 和 36-234 (吃碰杠等)，共 201 个动作
        self.head_interaction = nn.Sequential(
            nn.Linear(self.d_model*self.seq_len, 512), # 交互逻辑更复杂，给更多参数
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 201)
        )
        
        self._value_branch = nn.Sequential(
            nn.Linear(self.d_model*self.seq_len, 512),
            nn.BatchNorm1d(512),  # 增加BN
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 5. 参数初始化（补充Transformer特有层）
        self._init_weights()

    def _init_weights(self):
        """统一初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        # 设备对齐
        device = next(self.parameters()).device
        obs = input_dict["observation"].float().to(device)
        mask = input_dict["action_mask"].float().to(device)
        
        # 维度断言（调试用）
        assert obs.shape[1:] == (143, 4, 9), f"Obs shape error: {obs.shape}, expected (B,143,4,9)"
        assert mask.shape[1] == 235, f"Mask shape error: {mask.shape}, expected (B,235)"
        
        # 1. CNN嵌入
        x = self.conv_embed(obs)  # [B, 256, 4, 9]
        
        # 2. 重塑为序列 [B, C, H, W] -> [B, H*W, C]
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, 4, 9, 256]
        
        # 3. 位置编码
        x = x + self.row_embed + self.col_embed 
        x = x.reshape(b, -1, c)       # [B, 36, 256]
        x = x + self.pos_encoder  # 广播机制：[1,36,256] + [B,36,256]
        
        # 4. Transformer编码
        x = self.transformer_encoder(x)  # [B, 36, 256]
        
        # 5. 展平
        x = x.reshape(b, -1)  # [B, 36*256]

        # 6. 输出头（Multi-Head 拼接逻辑）
        # 分别计算两个头的输出
        out_play = self.head_play(x)          # [B, 34]
        out_int = self.head_interaction(x)    # [B, 201]

        # 重新拼装成 [B, 235] 的完整 logits
        # 索引映射依据 feature.py:
        # [0:2]   -> Pass, Hu (来自 head_interaction 的前2位)
        # [2:36]  -> Play (来自 head_play 的全部34位)
        # [36:235]-> Chi, Peng, Gang... (来自 head_interaction 的后199位)
        
        logits = torch.zeros(b, 235, device=device)
        logits[:, 0:2] = out_int[:, 0:2]
        logits[:, 2:36] = out_play
        logits[:, 36:] = out_int[:, 2:]
        
        # 安全的mask处理：替换log(0)为-1e9，避免inf
        inf_mask = torch.where(mask == 0, torch.tensor(-1e9, device=device), torch.tensor(0.0, device=device))
        masked_logits = logits + inf_mask
        
        # 7. Value分支
        value = self._value_branch(x)
        
        return masked_logits, value
    
class HandPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            nn.Conv2d(143, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        ]
        for _ in range(4):
            layers.append(ResidualBlock(128, 128))

        layers += [
            ResidualBlock(128, 64),
            ResidualBlock(64, 32),
            nn.Flatten()
        ]

        layers += nn.Sequential(
            nn.Linear(32 * 4 * 9, 512),
            nn.ReLU(True),
            nn.Linear(512, 102),
        )

        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        return self.net(obs)


class PreHandsModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 手牌预测子网
        self.hand_predictor = HandPredictor()

        tower_layers = [
            nn.Conv2d(155, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        ]
        num_res_blocks = 8
        for _ in range(num_res_blocks):
            tower_layers.append(ResidualBlock(256, 256))

        tower_layers += [
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 64),
            ResidualBlock(64, 32),
            nn.Flatten()
        ]
        self._tower = nn.Sequential(*tower_layers)

        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 9, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 235)
        )

        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 9, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # 精细初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 用于暴露最近一次预测（detach，避免被外部误用来反向传播）
        self.last_predicted_hands = None

    def forward(self, input_dict, mode = "total"):
        if mode == "decide":
            obs = input_dict["observation"].float()
            hands = input_dict["oppo_hands"].float()
            if len(hands.shape) == 3:
                hands = hands.unsqueeze(0)
            augmented = torch.cat([obs, hands], dim=1)

            hidden = self._tower(augmented)
            logits = self._logits(hidden)
            mask = input_dict["action_mask"].float()
            inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
            masked_logits = logits + inf_mask
            value = self._value_branch(hidden)
            return masked_logits, value
        elif mode == "predict":
            obs = input_dict["observation"].float()
            pred = self.hand_predictor(obs)
            self.last_predicted_hands = pred.detach()
            return pred
        elif mode == "total":
            obs = input_dict["observation"].float()
            pred = self.hand_predictor(obs)
            if len(pred.shape) == 3:
                pred = pred.unsqueeze(0)
                
            n = pred.shape[0]
            pred = pred.view(n, 3, 34)
            zeros = torch.zeros(n, 3, 2, device=pred.device, dtype=pred.dtype)
            pred = torch.cat([pred, zeros], dim=2)
            pred = pred.view(n, 3, 4, 9).unsqueeze(-1)
            k = torch.arange(4, device=pred.device).view(1,1,1,1,4)
            hands = torch.clamp(pred - k, min=0.0, max=1.0).permute(0,1,4,2,3).reshape(n, 12, 4, 9)

            if len(hands.shape) == 3:
                hands = hands.unsqueeze(0)
            augmented = torch.cat([obs, hands], dim=1)

            hidden = self._tower(augmented)
            logits = self._logits(hidden)
            mask = input_dict["action_mask"].float()
            inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
            masked_logits = logits + inf_mask
            value = self._value_branch(hidden)
            return masked_logits, value
