#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天云智能攻击框架 - 增强AI版
仅限合法授权使用

警告：未经授权使用此工具攻击他人系统是违法行为。
使用者需自行承担所有法律责任。
"""

import re
import cmd
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from transformers import BertModel, BertTokenizer
from torch import nn
from torch.utils.data import Dataset, DataLoader
from 思考模块 import HumanLikeThinker
from 自动写代码 import AutoCoder
from APT恶意代码 import APTModule
from 全自动化漏洞流程metasploit import MetaAutoPwn
from 量子模块 import HybridQuantumCrypto
from 硬件渗透模块 import HardwareC2
from 多层跳板加虚假攻击嫁祸 import FalseFlagAttackSimulator

# ==================== 法律声明 ====================
LEGAL_DISCLAIMER = """
使用条款:
1. 本工具仅限用于合法授权的安全测试和研究目的
2. 使用前必须获得目标系统的书面授权
3. 所有操作将被记录用于审计目的
4. 违反上述条款造成的后果由使用者自行承担
"""

print(LEGAL_DISCLAIMER)


# ==================== AI模型部分 ====================
class AttackRecommendationModel(nn.Module):
    """基于BERT的攻击策略推荐模型"""

    def __init__(self, num_classes=5):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)


class AttackDataset(Dataset):
    """攻击策略数据集"""

    def __init__(self, data_path: str):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = self._create_input_text(item)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'])
        }

    def _create_input_text(self, item):
        return f"""
目标IP: {item.get('ip', '未知')}
操作系统: {item.get('os', '未知')}
开放服务: {', '.join(item.get('services', []))}
已知漏洞: {', '.join(item.get('vulnerabilities', []))}
网络位置: {item.get('network_position', '未知')}
"""


class SkyCloudAI:
    """增强版AI决策引擎"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self._load_model()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.action_space = [
            "漏洞利用", "钓鱼攻击", "横向移动",
            "规避检测", "数据渗出"
        ]

    def _load_model(self) -> nn.Module:
        """加载预训练模型"""
        try:
            model = AttackRecommendationModel()
            model.load_state_dict(
                torch.load('models/attack_recommender.pth',
                           map_location=self.device)
            )
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}, 使用备用模型")
            return self._create_fallback_model()

    def _create_fallback_model(self) -> nn.Module:
        """创建备用模型"""
        model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.action_space))
        )
        return model.to(self.device)

    def recommend_attack(self, target_info: Dict) -> Dict:
        """
        为目标推荐攻击策略

        参数:
            target_info: 包含目标信息的字典，例如:
                {
                    'ip': '192.168.1.1',
                    'os': 'Windows 10',
                    'services': ['http', 'rdp'],
                    'vulnerabilities': ['CVE-2020-1472'],
                    'authorized': True  # 必须包含授权标志
                }

        返回:
            攻击建议字典，包含策略和置信度
        """
        if not target_info.get('authorized', False):
            raise ValueError("目标未授权 - 拒绝提供建议")

        inputs = self.tokenizer(
            self._create_input_text(target_info),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs)
            probs = torch.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, dim=1)
        top3 = torch.topk(probs, 3, dim=1)

        return {
            "recommendation": {
                "tactic": self.action_space[pred.item()],
                "confidence": confidence.item()
            },
            "alternatives": [
                {"tactic": self.action_space[i], "confidence": p.item()}
                for p, i in zip(top3.values[0], top3.indices[0])
            ],
            "model": "attack_recommender_v2",
            "timestamp": datetime.now().isoformat()
        }

    def _create_input_text(self, target_info: Dict) -> str:
        """创建模型输入文本"""
        return f"""
目标信息:
- IP地址: {target_info.get('ip', '未知')}
- 操作系统: {target_info.get('os', '未知')}
- 开放服务: {', '.join(target_info.get('services', []))}
- 已知漏洞: {', '.join(target_info.get('vulnerabilities', []))}
- 网络位置: {target_info.get('network_position', '未知')}
- 安全防护: {', '.join(target_info.get('defenses', ['未知']))}
"""


# ==================== 主框架 ====================
class SkyCloudCLI(cmd.Cmd):
    """天云交互式命令行界面"""

    prompt = "\n天云AI> "
    intro = """
==============================================
  天云AI攻击框架 v3.0 | 增强AI模式已激活
  输入 help 查看命令列表 | 输入 exit 退出
==============================================
""" + LEGAL_DISCLAIMER

    def __init__(self):
        super().__init__()
        self.ai_engine = SkyCloudAI()
        self.current_target = None
        self.session_log = []

        # 加载配置
        self.config = self._load_config()
        self._check_license()

    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open("config.json") as f:
                config = json.load(f)
                if not config.get("authorized", False):
                    raise ValueError("未授权配置")
                return config
        except Exception as e:
            print(f"⚠️ 配置加载失败: {e}")
            return {
                "authorized": False,
                "c2_server": None,
                "license_key": None
            }

    def _check_license(self):
        """检查许可证"""
        if not self.config.get("authorized", False):
            print("❌ 未检测到有效许可证，系统将在基础模式下运行")
            print("请联系供应商获取合法授权")

    def do_ai_scan(self, arg: str):
        """
        AI增强扫描
        示例: ai_scan 192.168.1.1 --os Windows --services http,rdp
        """
        if not self.config.get("authorized", False):
            print("❌ 此功能需要授权许可证")
            return

        args = self._parse_args(arg)
        target_info = {
            "ip": args.get("ip", ""),
            "os": args.get("os", "未知"),
            "services": args.get("services", "").split(","),
            "vulnerabilities": [],
            "authorized": True
        }

        print(f"🔍 AI正在分析 {target_info['ip']}...")
        try:
            recommendation = self.ai_engine.recommend_attack(target_info)
            self._display_recommendation(recommendation)
            self.current_target = target_info
        except Exception as e:
            print(f"❌ AI分析失败: {e}")

    def _parse_args(self, arg: str) -> Dict:
        """解析命令行参数"""
        args = {}
        ip_match = re.search(r"(\d+\.\d+\.\d+\.\d+)", arg)
        if ip_match:
            args["ip"] = ip_match.group(1)

        os_match = re.search(r"--os (\w+)", arg)
        if os_match:
            args["os"] = os_match.group(1)

        services_match = re.search(r"--services (\S+)", arg)
        if services_match:
            args["services"] = services_match.group(1)

        return args

    def _display_recommendation(self, rec: Dict):
        """显示AI推荐结果"""
        print("\n=== AI攻击策略推荐 ===")
        print(f"主策略: {rec['recommendation']['tactic']} "
              f"(置信度: {rec['recommendation']['confidence'] * 100:.1f}%)")

        print("\n备选策略:")
        for alt in rec['alternatives']:
            print(f"- {alt['tactic']} ({alt['confidence'] * 100:.1f}%)")

        print(f"\n模型版本: {rec['model']}")
        print(f"生成时间: {rec['timestamp']}")

        # 记录到会话日志
        self.session_log.append(rec)

    def do_exit(self, arg: str):
        """退出系统"""
        print("🛑 正在清理会话数据...")
        self._save_session_log()
        return True

    def _save_session_log(self):
        """保存会话日志"""
        try:
            filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "session_log": self.session_log,
                    "end_time": datetime.now().isoformat(),
                    "target": self.current_target
                }, f, indent=2)
            print(f"✅ 会话日志已保存到 {filename}")
        except Exception as e:
            print(f"❌ 日志保存失败: {e}")


if __name__ == "__main__":
    try:
        SkyCloudCLI().cmdloop()
    except KeyboardInterrupt:
        print("\n安全终止...")
    except Exception as e:
        print(f"❌ 系统错误: {e}")