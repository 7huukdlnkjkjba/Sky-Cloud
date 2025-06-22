import os
import subprocess
import time
from datetime import datetime

class AutoCoder:
    def __init__(self):
        self.template_dir = "templates"
        self.output_dir = "generated_code"
        self.setup_dirs()
        
    def setup_dirs(self):
        """创建必要的目录"""
        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_c_code(self, function_name, params, return_type, body):
        """生成C语言代码"""
        code = f"""#include <stdio.h>

{return_type} {function_name}({', '.join(params)}) {{
    {body}
    return 0;
}}

int main() {{
    // 自动生成的测试代码
    {function_name}();
    printf("程序执行成功!\\n");
    return 0;
}}
"""
        filename = f"{self.output_dir}/{function_name}.c"
        with open(filename, 'w') as f:
            f.write(code)
        return filename
    
    def generate_cpp_code(self, class_name, methods):
        """生成C++代码"""
        code = f"""#include <iostream>
using namespace std;

class {class_name} {{
public:
"""
        # 添加方法
        for method in methods:
            code += f"    {method['return_type']} {method['name']}({', '.join(method['params'])}) {{\n"
            code += f"        {method['body']}\n"
            code += "    }\n\n"
            
        code += f"""}};

int main() {{
    {class_name} obj;
    // 自动生成的测试代码
    cout << "C++程序执行成功!" << endl;
    return 0;
}}
"""
        filename = f"{self.output_dir}/{class_name}.cpp"
        with open(filename, 'w') as f:
            f.write(code)
        return filename
    
    def generate_java_code(self, class_name, methods):
        """生成Java代码"""
        code = f"""public class {class_name} {{
"""
        # 添加方法
        for method in methods:
            code += f"    public {method['return_type']} {method['name']}({', '.join(method['params'])}) {{\n"
            code += f"        {method['body']}\n"
            code += "    }\n\n"
            
        code += f"""    public static void main(String[] args) {{
        // 自动生成的测试代码
        System.out.println("Java程序执行成功!");
    }}
}}
"""
        filename = f"{self.output_dir}/{class_name}.java"
        with open(filename, 'w') as f:
            f.write(code)
        return filename
    
    def compile_code(self, filename):
        """编译生成的代码"""
        ext = os.path.splitext(filename)[1]
        output_name = os.path.splitext(os.path.basename(filename))[0]
        
        try:
            if ext == '.c':
                # 编译C代码
                cmd = f"gcc {filename} -o {self.output_dir}/{output_name}"
                subprocess.run(cmd, shell=True, check=True)
                return f"{self.output_dir}/{output_name}"
            elif ext == '.cpp':
                # 编译C++代码
                cmd = f"g++ {filename} -o {self.output_dir}/{output_name}"
                subprocess.run(cmd, shell=True, check=True)
                return f"{self.output_dir}/{output_name}"
            elif ext == '.java':
                # 编译Java代码
                cmd = f"javac {filename}"
                subprocess.run(cmd, shell=True, check=True)
                return f"{class_name}"
        except subprocess.CalledProcessError as e:
            print(f"编译失败: {e}")
            return None
    
    def run_code(self, executable, lang):
        """运行编译后的代码"""
        try:
            if lang == 'c' or lang == 'cpp':
                subprocess.run(f"./{executable}", shell=True, check=True)
            elif lang == 'java':
                # 对于Java，需要在输出目录运行
                original_dir = os.getcwd()
                os.chdir(self.output_dir)
                subprocess.run(f"java {executable}", shell=True, check=True)
                os.chdir(original_dir)
            print("程序执行成功!")
        except subprocess.CalledProcessError as e:
            print(f"执行失败: {e}")
    
    def auto_generate_and_compile(self, lang='c'):
        """自动生成并编译代码"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if lang == 'c':
            filename = self.generate_c_code(
                function_name=f"auto_func_{timestamp}",
                params=[],
                return_type="int",
                body='printf("这是自动生成的C函数!\\n");'
            )
            executable = self.compile_code(filename)
            if executable:
                self.run_code(executable, lang)
                
        elif lang == 'cpp':
            filename = self.generate_cpp_code(
                class_name=f"AutoClass_{timestamp}",
                methods=[{
                    'name': "printMessage",
                    'params': [],
                    'return_type': "void",
                    'body': 'cout << "这是自动生成的C++方法!" << endl;'
                }]
            )
            executable = self.compile_code(filename)
            if executable:
                self.run_code(executable, lang)
                
        elif lang == 'java':
            filename = self.generate_java_code(
                class_name=f"AutoClass_{timestamp}",
                methods=[{
                    'name': "printMessage",
                    'params': [],
                    'return_type': "void",
                    'body': 'System.out.println("这是自动生成的Java方法!");'
                }]
            )
            executable = self.compile_code(filename)
            if executable:
                self.run_code(executable, lang)
        
        else:
            print(f"不支持的语言: {lang}")

if __name__ == "__main__":
    coder = AutoCoder()
    
    print("自动代码生成与编译工具")
    print("1. 生成并编译C代码")
    print("2. 生成并编译C++代码")
    print("3. 生成并编译Java代码")
    
    choice = input("请选择(1/2/3): ")
    
    if choice == '1':
        coder.auto_generate_and_compile('c')
    elif choice == '2':
        coder.auto_generate_and_compile('cpp')
    elif choice == '3':
        coder.auto_generate_and_compile('java')
    else:
        print("无效选择")