import random
import time
from datetime import datetime, timedelta
import socket
import paramiko
import hashlib
import os


class AdvancedAttackSimulator:
    def __init__(self):
        # 模拟的C2服务器和跳板
        self.c2_servers = [
            {"ip": "203.0.113.10", "type": "C2", "location": "US"},
            {"ip": "198.51.100.20", "type": "Proxy", "location": "DE"},
            {"ip": "192.0.2.30", "type": "Relay", "location": "RU"}
        ]

        # 常见攻击工具指纹
        self.tool_fingerprints = {
            "Cobalt Strike": {
                "http_header": "Server: Apache/2.4.7 (Ubuntu)",
                "ssl_cert_hash": "a1b2c3d4e5f6",
                "behavior": "Beacon every 5 minutes"
            },
            "Metasploit": {
                "http_header": "Server: Apache/2.2.15 (CentOS)",
                "ssl_cert_hash": "x7y8z9a1b2c3",
                "behavior": "Randomized URI paths"
            }
        }

    def simulate_c2_communication(self, target_ip):
        """模拟C2通信"""
        print(f"\n[+] 模拟与C2服务器的通信: {target_ip}")

        # 随机选择C2服务器
        c2 = random.choice(self.c2_servers)
        print(f"  → 使用C2服务器: {c2['ip']} ({c2['location']})")

        # 模拟HTTP请求
        tool = random.choice(list(self.tool_fingerprints.keys()))
        fingerprint = self.tool_fingerprints[tool]
        print(f"  → 工具指纹: {tool}")
        print(f"     HTTP Header: {fingerprint['http_header']}")
        print(f"     SSL Cert Hash: {fingerprint['ssl_cert_hash']}")

        # 模拟心跳通信
        for i in range(3):
            time.sleep(random.uniform(2, 5))
            print(f"  → 心跳信号 #{i + 1} 发送到 {c2['ip']}")

    def generate_malicious_file(self, file_type="exe"):
        """生成模拟恶意文件"""
        print("\n[+] 生成模拟恶意文件")

        # 创建文件哈希
        random_data = os.urandom(256)
        file_hash = hashlib.sha256(random_data).hexdigest()

        # 模拟常见恶意文件特征
        if file_type == "exe":
            print(f"  → 生成PE文件 (SHA256: {file_hash[:16]}...)")
            print("     PE特征: 加壳区段, 可疑导入表")
        elif file_type == "doc":
            print(f"  → 生成恶意文档 (SHA256: {file_hash[:16]}...)")
            print("     包含宏代码: AutoOpen()执行powershell")

        return file_hash

    def simulate_lateral_movement(self, target_network):
        """模拟横向移动"""
        print(f"\n[+] 模拟横向移动在 {target_network}")

        # 模拟SMB传播
        print("  → 尝试通过SMB传播")
        time.sleep(1)

        # 模拟凭证窃取
        print("  → 使用Mimikatz提取凭证")
        print("     Administrator: NTLM hash [a1b2c3d4e5f6]")

        # 模拟PsExec执行
        print("  → 通过PsExec在内部主机执行payload")
        for i in range(1, 4):
            print(f"     感染内部主机: 192.168.1.{i}")


# 使用示例
if __name__ == "__main__":
    simulator = AdvancedAttackSimulator()

    # 模拟C2通信
    simulator.simulate_c2_communication("10.0.0.100")

    # 生成恶意文件
    simulator.generate_malicious_file("doc")

    # 模拟横向移动
    simulator.simulate_lateral_movement("192.168.1.0/24")