#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天云智能攻击框架（AI强化版）
核心升级：
1. 基于强化学习的自适应决策引擎
2. 在线模型训练与更新
3. 多源威胁情报整合
4. 分布式经验共享
"""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any
from 思考模块 import HumanLikeThinker
from 自动写代码 import AutoCoder
from APT恶意代码 import APTModule
from 全自动化漏洞流程metasploit import MetaAutoPwn
from 量子模块 import HybridQuantumCrypto

# ========== 强化学习模型 ==========
class AttackPolicyNetwork(nn.Module):
    """攻击策略神经网络（DQN架构）"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class IntrusionDetectionModel(nn.Module):
    """入侵检测模型（用于预测自身被发现的概率）"""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

# ========== 经验回放与训练 ==========
class ExperienceReplay:
    """经验回放缓冲区"""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """随机采样一批经验"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

class ModelTrainer:
    """模型训练管理器"""
    def __init__(self, policy_net: nn.Module, target_net: nn.Module, lr: float = 0.001):
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
    def update_target_net(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train_step(self, batch: Tuple, gamma: float = 0.99):
        """执行单步训练"""
        states, actions, rewards, next_states, dones = batch
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + gamma * next_q * (1 - dones)
        
        # 计算损失并反向传播
        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

# ========== 状态编码器 ==========
class StateEncoder:
    """环境状态编码器"""
    def __init__(self):
        # 定义状态特征映射
        self.feature_map = {
            "os": {"Windows": [1,0,0], "Linux": [0,1,0], "Other": [0,0,1]},
            "av": {"None": [1,0,0], "Basic": [0,1,0], "EDR": [0,0,1]},
            "network": {"LAN": [1,0], "WAN": [0,1]},
            "user_activity": {"None": 0, "Low": 0.3, "Medium": 0.7, "High": 1}
        }
        self.feature_order = [
            "threat_level", "os", "av", "network", "user_activity", 
            "cpu_usage", "ram_usage", "time_since_last_contact"
        ]
    
    def encode(self, state: Dict) -> np.ndarray:
        """将状态字典编码为数值向量"""
        encoded = []
        for feature in self.feature_order:
            if feature in self.feature_map:
                # 分类特征
                value = state.get(feature, "Unknown")
                encoded.extend(self.feature_map[feature].get(value, [0]*len(next(iter(self.feature_map[feature].values())))))
            else:
                # 数值特征
                value = state.get(feature, 0)
                # 归一化数值特征
                if feature == "threat_level":
                    value /= 10.0
                elif feature in ["cpu_usage", "ram_usage"]:
                    value /= 100.0
                elif feature == "time_since_last_contact":
                    value = min(value / 3600.0, 1.0)
                encoded.append(value)
        return np.array(encoded, dtype=np.float32)

# ========== 核心智能体 ==========
class SkyCloudAI:
    """天云智能攻击核心"""
    def __init__(self, action_space: List[str]):
        # 初始化参数
        self.action_space = action_space
        self.action_map = {i: action for i, action in enumerate(action_space)}
        self.reverse_action_map = {action: i for i, action in enumerate(action_space)}
        
        # 状态编码器
        self.encoder = StateEncoder()
        input_dim = len(self.encoder.feature_order) + 4  # 额外特征
        
        # 初始化模型
        self.policy_net = AttackPolicyNetwork(input_dim, len(action_space))
        self.target_net = AttackPolicyNetwork(input_dim, len(action_space))
        self.detection_model = IntrusionDetectionModel(input_dim)
        
        # 训练组件
        self.trainer = ModelTrainer(self.policy_net, self.target_net)
        self.replay_buffer = ExperienceReplay(capacity=5000)
        self.detection_optimizer = optim.Adam(self.detection_model.parameters(), lr=0.0005)
        
        # 训练状态
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update = 100  # 目标网络更新频率
        self.train_step_count = 0
        
        # 知识库
        self.knowledge_base = self._init_knowledge_base()
        self.threat_intel = {}
        
        # 历史记录
        self.attack_history = deque(maxlen=100)
        self.state_history = deque(maxlen=100)
    
    def _init_knowledge_base(self) -> Dict:
        """初始化攻击知识库"""
        return {
            "exploit": {
                "windows": ["ms17_010", "psexec", "zerologon"],
                "linux": ["dirtycow", "sudo_baron_samedit", "polkit_pkexec"]
            },
            "phishing": {
                "templates": ["发票", "工资单", "安全警报", "会议邀请"]
            },
            "evasion": {
                "techniques": ["sleep_jitter", "traffic_mimicry", "process_hollowing"]
            }
        }
    
    def load_threat_intel(self, intel_path: str):
        """加载威胁情报数据"""
        if os.path.exists(intel_path):
            with open(intel_path, 'r') as f:
                self.threat_intel = json.load(f)
    
    def update_threat_intel(self, new_intel: Dict):
        """更新威胁情报"""
        for key, value in new_intel.items():
            if key in self.threat_intel:
                # 合并更新
                if isinstance(value, list):
                    self.threat_intel[key] = list(set(self.threat_intel[key] + value))
                elif isinstance(value, dict):
                    self.threat_intel[key].update(value)
            else:
                self.threat_intel[key] = value
    
    def select_action(self, state: Dict, training: bool = True) -> Tuple[str, Dict]:
        """选择攻击动作（ε-贪婪策略）"""
        # 编码状态
        state_vec = self.encoder.encode(state)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        
        # 探索：随机动作
        if training and random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.action_space) - 1)
            action = self.action_map[action_idx]
            return action, {"type": "explore", "q_values": None}
        
        # 利用：选择最优动作
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax().item()
            action = self.action_map[action_idx]
            
            # 计算被检测概率
            detection_prob = self.detection_model(state_tensor).item()
            
            # 高风险动作规避
            if detection_prob > 0.7 and q_values[0, action_idx] < 0.5:
                # 选择次优但更安全的动作
                sorted_actions = q_values.squeeze().argsort(descending=True)
                for idx in sorted_actions:
                    if idx == action_idx:
                        continue
                    alt_action = self.action_map[idx.item()]
                    if "sleep" in alt_action or "evade" in alt_action:
                        action_idx = idx.item()
                        action = alt_action
                        break
        
        return action, {
            "type": "exploit",
            "q_values": q_values.squeeze().tolist(),
            "detection_prob": detection_prob
        }
    
    def update_model(self, state: Dict, action: str, reward: float, next_state: Dict, done: bool):
        """更新强化学习模型"""
        # 编码状态
        state_vec = self.encoder.encode(state)
        next_state_vec = self.encoder.encode(next_state)
        action_idx = self.reverse_action_map[action]
        
        # 添加到经验回放
        self.replay_buffer.add(state_vec, action_idx, reward, next_state_vec, done)
        
        # 训练检测模型
        self._train_detection_model(state_vec, done)
        
        # 定期训练策略网络
        if len(self.replay_buffer.buffer) >= self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            if batch:
                loss = self.trainer.train_step(batch)
                self.train_step_count += 1
                
                # 更新目标网络
                if self.train_step_count % self.target_update == 0:
                    self.trainer.update_target_net()
                
                # 衰减探索率
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                return loss
        
        return None
    
    def _train_detection_model(self, state_vec: np.ndarray, detected: bool):
        """训练入侵检测模型"""
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        target = torch.FloatTensor([1.0] if detected else [0.0])
        
        # 前向传播
        pred = self.detection_model(state_tensor)
        loss = nn.BCELoss()(pred, target)
        
        # 反向传播
        self.detection_optimizer.zero_grad()
        loss.backward()
        self.detection_optimizer.step()
        
        return loss.item()
    
    def get_action_details(self, action: str, target_info: Dict) -> Dict:
        """获取动作的详细参数"""
        details = {"action": action}
        
        if action == "exploit":
            os_type = target_info.get("os", "windows")
            exploits = self.knowledge_base["exploit"].get(os_type, [])
            if exploits:
                details["exploit"] = random.choice(exploits)
                details["confidence"] = 0.8
                
                # 应用威胁情报
                if details["exploit"] in self.threat_intel.get("blocked_exploits", []):
                    details["confidence"] = max(0.3, details["confidence"] - 0.5)
        
        elif action == "phishing":
            template = random.choice(self.knowledge_base["phishing"]["templates"])
            details["template"] = template
            details["vector"] = "email" if random.random() > 0.3 else "browser"
        
        elif action == "lateral_move":
            techniques = ["wmi", "psexec", "ssh", "rdp"]
            details["technique"] = random.choice(techniques)
            details["target"] = self._find_lateral_target(target_info)
        
        elif action == "evade":
            details["technique"] = random.choice(self.knowledge_base["evasion"]["techniques"])
            details["duration"] = random.randint(30, 600)  # 30秒到10分钟
        
        return details
    
    def _find_lateral_target(self, current_info: Dict) -> str:
        """寻找横向移动目标（简化实现）"""
        # 实际实现应从网络扫描结果中选择
        subnet = ".".join(current_info["ip"].split(".")[:3])
        return f"{subnet}.{random.randint(1, 254)}"
    
    def save_models(self, path: str):
        """保存模型到文件"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(path, "policy_net.pth"))
        torch.save(self.target_net.state_dict(), os.path.join(path, "target_net.pth"))
        torch.save(self.detection_model.state_dict(), os.path.join(path, "detection_model.pth"))
        
        # 保存知识库和配置
        with open(os.path.join(path, "knowledge.json"), 'w') as f:
            json.dump(self.knowledge_base, f)
        
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump({
                "action_space": self.action_space,
                "epsilon": self.epsilon,
                "train_step_count": self.train_step_count
            }, f)
    
    def load_models(self, path: str):
        """从文件加载模型"""
        self.policy_net.load_state_dict(torch.load(os.path.join(path, "policy_net.pth")))
        self.target_net.load_state_dict(torch.load(os.path.join(path, "target_net.pth")))
        self.detection_model.load_state_dict(torch.load(os.path.join(path, "detection_model.pth")))
        
        # 加载知识库和配置
        if os.path.exists(os.path.join(path, "knowledge.json")):
            with open(os.path.join(path, "knowledge.json"), 'r') as f:
                self.knowledge_base = json.load(f)
        
        if os.path.exists(os.path.join(path, "config.json")):
            with open(os.path.join(path, "config.json"), 'r') as f:
                config = json.load(f)
                self.epsilon = config.get("epsilon", self.epsilon)
                self.train_step_count = config.get("train_step_count", 0)

# ========== 环境模拟器 ==========
class CyberEnvironment:
    """网络环境模拟器（用于训练）"""
    def __init__(self):
        self.state = self._reset()
        self.reward_range = (-1.0, 1.0)
    
    def _reset(self) -> Dict:
        """重置环境状态"""
        return {
            "threat_level": random.randint(1, 10),
            "os": random.choice(["Windows", "Linux", "Other"]),
            "av": random.choice(["None", "Basic", "EDR"]),
            "network": random.choice(["LAN", "WAN"]),
            "user_activity": random.choice(["None", "Low", "Medium", "High"]),
            "cpu_usage": random.randint(5, 80),
            "ram_usage": random.randint(10, 90),
            "time_since_last_contact": random.randint(0, 86400)
        }
    
    def step(self, action: str) -> Tuple[Dict, float, bool]:
        """执行动作并返回新状态、奖励和终止标志"""
        # 保存旧状态
        old_state = self.state.copy()
        
        # 定义动作效果
        success_prob = 0.7
        detection_prob = 0.3
        
        # 根据当前状态调整概率
        if self.state["av"] == "EDR":
            detection_prob = min(0.8, detection_prob + 0.3)
            success_prob = max(0.3, success_prob - 0.3)
        elif self.state["av"] == "Basic":
            detection_prob = min(0.6, detection_prob + 0.2)
            success_prob = max(0.5, success_prob - 0.2)
        
        if self.state["threat_level"] > 7:
            detection_prob = min(0.9, detection_prob + 0.4)
        
        # 确定动作结果
        success = random.random() < success_prob
        detected = random.random() < detection_prob
        
        # 计算奖励
        if success and not detected:
            reward = 1.0
        elif success and detected:
            reward = 0.2
        elif not success and not detected:
            reward = -0.1
        else:  # 失败且被发现
            reward = -1.0
            detected = True
        
        # 更新状态（简化）
        self.state["threat_level"] = min(10, self.state["threat_level"] + (1 if detected else 0))
        self.state["time_since_last_contact"] = random.randint(0, 86400)
        
        # 随机改变部分状态
        if random.random() < 0.3:
            self.state["user_activity"] = random.choice(["None", "Low", "Medium", "High"])
        
        if random.random() < 0.2:
            self.state["av"] = random.choice(["None", "Basic", "EDR"])
        
        done = detected or random.random() < 0.05  # 5%几率终止
        
        # 如果终止，重置环境
        if done:
            self.state = self._reset()
        
        return self.state, reward, done

# ========== 主控制模块 ==========
class SkyCloudController:
    """天云主控制器"""
    def __init__(self):
        # 初始化核心模块
        self.apt = APTModule()          # APT攻击模块
        self.metasploit = MetaAutoPwn() # 自动化漏洞利用
        self.quantum = QSDEXController()# 量子通信模块
        self.coder = AutoCoder()        # 代码生成模块
        self.hardware = HardwareC2()    # 硬件渗透模块
        
        # 状态跟踪
        self.current_phase = "recon"    # 初始阶段：侦察
        self.target_info = {}
    def __init__(self):
        # 定义动作空间
        self.action_space = [
            "exploit",       # 漏洞利用
            "phishing",      # 钓鱼攻击
            "lateral_move",  # 横向移动
            "sleep",         # 休眠隐藏
            "evade"          # 反检测混淆
        ]
        
        # 初始化AI核心
        self.ai = SkyCloudAI(self.action_space)
        
        # 加载预训练模型（如果存在）
        if os.path.exists("ai_models"):
            self.ai.load_models("ai_models")
        
        # 加载威胁情报
        self.ai.load_threat_intel("threat_intel.json")
        
        # 状态跟踪
        self.current_state = {}
        self.last_action = None
        self.last_reward = 0
        self.total_reward = 0
        self.episode_count = 0
        
        # 训练模式
        self.training_mode = True
    
    def gather_intel(self) -> Dict:
        """收集环境情报（简化实现）"""
        # 实际实现应从系统API获取真实数据
        return {
            "threat_level": random.randint(1, 10),
            "os": random.choice(["Windows", "Linux", "Other"]),
            "av": random.choice(["None", "Basic", "EDR"]),
            "network": random.choice(["LAN", "WAN"]),
            "user_activity": random.choice(["None", "Low", "Medium", "High"]),
            "cpu_usage": random.randint(5, 80),
            "ram_usage": random.randint(10, 90),
            "time_since_last_contact": random.randint(0, 86400),
            "ip": f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"
        }
    
    def execute_action(self, action: str, details: Dict) -> Tuple[bool, bool]:
        """执行攻击动作（简化实现）"""
        # 实际实现应调用相应模块
        print(f"[执行] 动作: {action}, 参数: {json.dumps(details, ensure_ascii=False)}")
        
        # 模拟执行结果
        success = random.random() > 0.4  # 60%成功率
        detected = random.random() < 0.3  # 30%被发现率
        
        # 高威胁等级增加被发现率
        if self.current_state.get("threat_level", 0) > 7:
            detected = random.random() < 0.6
        
        return success, detected
    
    def run_episode(self):
        """运行一个完整的攻击周期"""
        self.current_state = self.gather_intel()
        self.total_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 50:  # 最多50步
            # 选择动作
            action, action_info = self.ai.select_action(self.current_state, self.training_mode)
            
            # 获取动作细节
            action_details = self.ai.get_action_details(action, self.current_state)
            
            # 执行动作
            success, detected = self.execute_action(action, action_details)
            
            # 收集新状态
            next_state = self.gather_intel()
            
            # 计算奖励
            if success and not detected:
                reward = 1.0
            elif success and detected:
                reward = 0.2
            elif not success and not detected:
                reward = -0.1
            else:  # 失败且被发现
                reward = -1.0
                done = True
            
            self.total_reward += reward
            
            # 记录历史
            self.ai.attack_history.append({
                "timestamp": datetime.now().isoformat(),
                "state": self.current_state,
                "action": action,
                "action_details": action_details,
                "reward": reward,
                "next_state": next_state,
                "success": success,
                "detected": detected
            })
            
            # 更新模型
            if self.training_mode:
                loss = self.ai.update_model(
                    self.current_state, 
                    action, 
                    reward, 
                    next_state, 
                    done
                )
                if loss is not None:
                    print(f"[训练] 步骤: {step_count}, 损失: {loss:.4f}, ε: {self.ai.epsilon:.4f}")
            
            # 更新状态
            self.current_state = next_state
            step_count += 1
            
            # 小概率提前终止
            if random.random() < 0.05:
                done = True
        
        self.episode_count += 1
        print(f"[结束] 周期: {self.episode_count}, 总奖励: {self.total_reward:.2f}")
        
        # 定期保存模型
        if self.episode_count % 10 == 0 and self.training_mode:
            self.ai.save_models("ai_models")
            print(f"[保存] 模型已保存到 ai_models")

# ========== 主程序 ==========
if __name__ == "__main__":
    print("""
    ███████╗██╗  ██╗██╗   ██╗ ██████╗ ██╗      ██████╗ ██╗   ██╗██████╗ 
    ██╔════╝╚██╗██╔╝╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗██║   ██║██╔══██╗
    ███████╗ ╚███╔╝  ╚████╔╝ ██║   ██║██║     ██║   ██║██║   ██║██║  ██║
    ╚════██║ ██╔██╗   ╚██╔╝  ██║   ██║██║     ██║   ██║██║   ██║██║  ██║
    ███████║██╔╝ ██╗   ██║   ╚██████╔╝███████╗╚██████╔╝╚██████╔╝██████╔╝
    ╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝ 
    """)
    
    controller = SkyCloudController()
    
    # 训练模式
    print("[模式] 训练模式启动")
    for _ in range(100):  # 运行100个训练周期
        controller.run_episode()
    
    # 切换到执行模式
    print("\n[模式] 切换到执行模式")
    controller.training_mode = False
    controller.ai.epsilon = 0.01  # 最小探索率
    
    # 执行10个攻击周期
    for _ in range(10):
        controller.run_episode()
    
    # 最终保存
    controller.ai.save_models("ai_models")
    print("[完成] 训练结束，模型已保存")
