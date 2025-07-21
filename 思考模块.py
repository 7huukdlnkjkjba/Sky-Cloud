import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import random


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


# --- 恶意代码进化器 ---
class MaliciousCodeEvolver:
    def __init__(self, target_model, population_size=10, mutation_rate=0.1):
        self.target_model = target_model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [torch.randn(128) for _ in range(population_size)]

    def mutate(self, code):
        # 添加随机突变
        mask = torch.rand_like(code) < self.mutation_rate
        noise = torch.randn_like(code) * 0.2
        return code + mask * noise

    def evaluate_fitness(self, code):
        # 评估代码的对抗性效果
        with torch.no_grad():
            output = self.target_model(code.unsqueeze(0))
            # 假设我们希望最大化模型的输出(欺骗模型)
            fitness = output.mean().item()
        return fitness

    def evolve(self, generations=5):
        for _ in range(generations):
            # 评估种群中每个个体的适应度
            fitness = [self.evaluate_fitness(code) for code in self.population]

            # 选择前50%的个体
            sorted_pop = [x for _, x in sorted(zip(fitness, self.population), reverse=True)]
            top_half = sorted_pop[:self.population_size // 2]

            # 通过变异和交叉生成新一代
            new_pop = []
            for parent in top_half:
                # 直接复制父代
                new_pop.append(deepcopy(parent))
                # 变异父代
                new_pop.append(self.mutate(deepcopy(parent)))

            self.population = new_pop

        # 返回最具对抗性的代码
        fitness = [self.evaluate_fitness(code) for code in self.population]
        return self.population[np.argmax(fitness)]


# --- 对抗训练增强的决策模型 ---
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
        self.evolver = MaliciousCodeEvolver(self.base_model)  # 添加进化器

    def forward(self, x):
        adv_score = self.adversarial_filter(x)
        if adv_score > 0.7:  # 检测到对抗样本
            x = x + torch.randn_like(x) * 0.1  # 加入噪声破坏对抗样本
            # 使用进化器生成对抗样本用于训练
            adversarial_sample = self.evolver.evolve()
            self.base_model.train()
            self.base_model(adversarial_sample.unsqueeze(0))
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

    # 测试恶意代码进化
    evolver = MaliciousCodeEvolver(decision_maker)
    malicious_code = evolver.evolve(generations=5)
    print("进化后的恶意代码形状:", malicious_code.shape)
    print("恶意代码效果:", decision_maker(malicious_code.unsqueeze(0)).item())