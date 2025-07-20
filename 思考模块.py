import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 注意力机制 ---
class HumanAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.working_memory = torch.zeros(1, dim)

    def forward(self, x):
        self.working_memory = 0.9 * self.working_memory + 0.1 * x
        q = self.query(self.working_memory)
        k = self.key(x)
        v = self.value(x)
        attn = F.softmax(q @ k.T / np.sqrt(k.size(-1)), dim=-1)
        return attn @ v


class AdversarialTrainingWrapper(nn.Module):
    """对抗训练增强的决策模型"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.adversarial_filter = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()  # 输出对抗样本置信度
        )

    def forward(self, x):
        adv_score = self.adversarial_filter(x)
        if adv_score > 0.7:  # 检测到对抗样本
            x = x + torch.randn_like(x) * 0.1  # 加入噪声破坏对抗样本
        return self.base_model(x)


# --- 决策模型 ---
class CognitiveDecisionMaker(nn.Module):
    def __init__(self):
        super().__init__()
        self.system1 = nn.LSTM(128, 64, bidirectional=True)
        self.system2 = nn.Transformer(d_model=128, nhead=8)
        self.emotional_bias = nn.Parameter(torch.zeros(128))

    def forward(self, inputs):
        fast_out, _ = self.system1(inputs.unsqueeze(0))
        slow_out = self.system2(inputs.unsqueeze(0), inputs.unsqueeze(0))
        combined = 0.7 * slow_out + 0.3 * fast_out + self.emotional_bias
        return torch.sigmoid(combined)

# --- 测试 ---
if __name__ == "__main__":
    attention = HumanAttention(128)
    decision_maker = CognitiveDecisionMaker()
    
    # 模拟输入（128维向量）
    x = torch.randn(128)
    
    # 注意力处理
    attended = attention(x)
    print("注意力输出形状:", attended.shape)  # 应为 [1, 128]

    # 修改原决策流程
    decision_maker = CognitiveDecisionMaker()
    hardened_decision_maker = AdversarialTrainingWrapper(decision_maker)
