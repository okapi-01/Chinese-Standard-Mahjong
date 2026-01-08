import torch
from torch import nn

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
    def __init__(self):
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