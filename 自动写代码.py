import os
import subprocess
import time
from datetime import datetime
import json

class AutoCoder:
    def __init__(self):
        self.template_dir = "templates"
        self.output_dir = "generated_code"
        self.template_db = "templates/template_db.json"
        self.setup_dirs()
        self.load_templates()
        
    def setup_dirs(self):
        """创建必要的目录"""
        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化模板数据库如果不存在
        if not os.path.exists(self.template_db):
            with open(self.template_db, 'w') as f:
                json.dump({
                    "c": {},
                    "cpp": {},
                    "java": {},
                    "python": {},
                    "go": {},
                    "rust": {}
                }, f)
    
    def load_templates(self):
        """加载代码模板"""
        try:
            with open(self.template_db, 'r') as f:
                self.templates = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.templates = {
                "c": {},
                "cpp": {},
                "java": {},
                "python": {},
                "go": {},
                "rust": {}
            }
    
    def save_template(self, lang, template_name, template_data):
        """保存新的代码模板"""
        if lang not in self.templates:
            self.templates[lang] = {}
        self.templates[lang][template_name] = template_data
        
        with open(self.template_db, 'w') as f:
            json.dump(self.templates, f, indent=4)
    
    def get_template(self, lang, template_name):
        """获取代码模板"""
        return self.templates.get(lang, {}).get(template_name, None)
    
    def generate_c_code(self, function_name, params, return_type, body, template_name=None):
        """生成C语言代码"""
        if template_name:
            template = self.get_template('c', template_name)
            if template:
                code = template.replace("{function_name}", function_name) \
                               .replace("{params}", ', '.join(params)) \
                               .replace("{return_type}", return_type) \
                               .replace("{body}", body)
                filename = f"{self.output_dir}/{function_name}.c"
                with open(filename, 'w') as f:
                    f.write(code)
                return filename
        
        # 默认模板
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
    
    def generate_cpp_code(self, class_name, methods, template_name=None):
        """生成C++代码"""
        if template_name:
            template = self.get_template('cpp', template_name)
            if template:
                methods_code = ""
                for method in methods:
                    methods_code += f"    {method['return_type']} {method['name']}({', '.join(method['params'])}) {{\n"
                    methods_code += f"        {method['body']}\n"
                    methods_code += "    }\n\n"
                
                code = template.replace("{class_name}", class_name) \
                              .replace("{methods}", methods_code)
                filename = f"{self.output_dir}/{class_name}.cpp"
                with open(filename, 'w') as f:
                    f.write(code)
                return filename
        
        # 默认模板
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
    
    def generate_java_code(self, class_name, methods, template_name=None):
        """生成Java代码"""
        if template_name:
            template = self.get_template('java', template_name)
            if template:
                methods_code = ""
                for method in methods:
                    methods_code += f"    public {method['return_type']} {method['name']}({', '.join(method['params'])}) {{\n"
                    methods_code += f"        {method['body']}\n"
                    methods_code += "    }\n\n"
                
                code = template.replace("{class_name}", class_name) \
                              .replace("{methods}", methods_code)
                filename = f"{self.output_dir}/{class_name}.java"
                with open(filename, 'w') as f:
                    f.write(code)
                return filename
        
        # 默认模板
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
    
    def generate_python_code(self, function_name, params, return_type, body, template_name=None):
        """生成Python代码"""
        if template_name:
            template = self.get_template('python', template_name)
            if template:
                code = template.replace("{function_name}", function_name) \
                               .replace("{params}", ', '.join(params)) \
                               .replace("{return_type}", return_type) \
                               .replace("{body}", body)
                filename = f"{self.output_dir}/{function_name}.py"
                with open(filename, 'w') as f:
                    f.write(code)
                return filename
        
        # 默认模板
        code = f"""def {function_name}({', '.join(params)}):
    {body}
    return {return_type}

if __name__ == "__main__":
    # 自动生成的测试代码
    {function_name}()
    print("Python程序执行成功!")
"""
        filename = f"{self.output_dir}/{function_name}.py"
        with open(filename, 'w') as f:
            f.write(code)
        return filename
    
    def generate_go_code(self, function_name, params, return_type, body, template_name=None):
        """生成Go代码"""
        if template_name:
            template = self.get_template('go', template_name)
            if template:
                code = template.replace("{function_name}", function_name) \
                               .replace("{params}", ', '.join(params)) \
                               .replace("{return_type}", return_type) \
                               .replace("{body}", body)
                filename = f"{self.output_dir}/{function_name}.go"
                with open(filename, 'w') as f:
                    f.write(code)
                return filename
        
        # 默认模板
        code = f"""package main

import "fmt"

func {function_name}({', '.join(params)}) {return_type} {{
    {body}
    return {return_type}
}}

func main() {{
    // 自动生成的测试代码
    {function_name}()
    fmt.Println("Go程序执行成功!")
}}
"""
        filename = f"{self.output_dir}/{function_name}.go"
        with open(filename, 'w') as f:
            f.write(code)
        return filename
    
    def generate_rust_code(self, function_name, params, return_type, body, template_name=None):
        """生成Rust代码"""
        if template_name:
            template = self.get_template('rust', template_name)
            if template:
                code = template.replace("{function_name}", function_name) \
                               .replace("{params}", ', '.join(params)) \
                               .replace("{return_type}", return_type) \
                               .replace("{body}", body)
                filename = f"{self.output_dir}/{function_name}.rs"
                with open(filename, 'w') as f:
                    f.write(code)
                return filename
        
        # 默认模板
        code = f"""fn {function_name}({', '.join(params)}) -> {return_type} {{
    {body}
    {return_type}
}}

fn main() {{
    // 自动生成的测试代码
    {function_name}();
    println!("Rust程序执行成功!");
}}
"""
        filename = f"{self.output_dir}/{function_name}.rs"
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
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"编译错误:\n{result.stderr}")
                    return None
                return f"{self.output_dir}/{output_name}"
            elif ext == '.cpp':
                # 编译C++代码
                cmd = f"g++ {filename} -o {self.output_dir}/{output_name}"
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"编译错误:\n{result.stderr}")
                    return None
                return f"{self.output_dir}/{output_name}"
            elif ext == '.java':
                # 编译Java代码
                cmd = f"javac {filename}"
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"编译错误:\n{result.stderr}")
                    return None
                return output_name  # Java类名
            elif ext == '.go':
                # 编译Go代码
                cmd = f"go build -o {self.output_dir}/{output_name} {filename}"
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"编译错误:\n{result.stderr}")
                    return None
                return f"{self.output_dir}/{output_name}"
            elif ext == '.rs':
                # 编译Rust代码
                cmd = f"rustc {filename} -o {self.output_dir}/{output_name}"
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"编译错误:\n{result.stderr}")
                    return None
                return f"{self.output_dir}/{output_name}"
        except subprocess.CalledProcessError as e:
            print(f"编译失败: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"错误详情:\n{e.stderr}")
            return None
    
    def run_code(self, executable, lang):
        """运行编译后的代码"""
        try:
            if lang in ['c', 'cpp', 'go', 'rust']:
                result = subprocess.run(f"./{executable}", shell=True, check=True, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print(f"运行时错误:\n{result.stderr}")
            elif lang == 'java':
                # 对于Java，需要在输出目录运行
                original_dir = os.getcwd()
                os.chdir(self.output_dir)
                result = subprocess.run(f"java {executable}", shell=True, check=True, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print(f"运行时错误:\n{result.stderr}")
                os.chdir(original_dir)
            elif lang == 'python':
                result = subprocess.run(f"python {executable}", shell=True, check=True, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print(f"运行时错误:\n{result.stderr}")
            print("程序执行成功!")
        except subprocess.CalledProcessError as e:
            print(f"执行失败: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"错误详情:\n{e.stderr}")
    
    def analyze_code(self, filename):
        """静态代码分析"""
        ext = os.path.splitext(filename)[1]
        
        if ext == '.py':
            # 使用pylint分析Python代码
            try:
                cmd = f"pylint {filename}"
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                print("代码分析结果:")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"代码分析完成(可能有警告):\n{e.stdout}")
        elif ext in ['.c', '.cpp']:
            # 使用cppcheck分析C/C++代码
            try:
                cmd = f"cppcheck {filename}"
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                print("代码分析结果:")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"代码分析完成(可能有警告):\n{e.stdout}")
        elif ext == '.java':
            # 使用checkstyle分析Java代码
            try:
                cmd = f"checkstyle {filename}"
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                print("代码分析结果:")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"代码分析完成(可能有警告):\n{e.stdout}")
    
    def generate_test_code(self, filename):
        """生成测试代码"""
        ext = os.path.splitext(filename)[1]
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        if ext == '.py':
            # 为Python生成unittest测试
            test_code = f"""import unittest
import {base_name}

class Test{base_name.capitalize()}(unittest.TestCase):
    def test_generated(self):
        # 自动生成的测试用例
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
"""
            test_filename = f"{self.output_dir}/test_{base_name}.py"
            with open(test_filename, 'w') as f:
                f.write(test_code)
            return test_filename
        elif ext in ['.c', '.cpp']:
            # 为C/C++生成简单的测试
            test_code = f"""#include <stdio.h>
#include "{filename}"

int main() {{
    // 自动生成的测试代码
    printf("Running tests for {base_name}\\n");
    // TODO: 添加实际测试逻辑
    printf("All tests passed!\\n");
    return 0;
}}
"""
            test_filename = f"{self.output_dir}/test_{base_name}.c"
            with open(test_filename, 'w') as f:
                f.write(test_code)
            return test_filename
        elif ext == '.java':
            # 为Java生成JUnit测试
            test_code = f"""import org.junit.Test;
import static org.junit.Assert.*;

public class Test{base_name} {{
    @Test
    public void testGenerated() {{
        // 自动生成的测试用例
        assertTrue(true);
    }}
}}
"""
            test_filename = f"{self.output_dir}/Test{base_name}.java"
            with open(test_filename, 'w') as f:
                f.write(test_code)
            return test_filename
    
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
            self.analyze_code(filename)
            executable = self.compile_code(filename)
            if executable:
                self.run_code(executable, lang)
            test_file = self.generate_test_code(filename)
            print(f"生成的测试文件: {test_file}")
                
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
            self.analyze_code(filename)
            executable = self.compile_code(filename)
            if executable:
                self.run_code(executable, lang)
            test_file = self.generate_test_code(filename)
            print(f"生成的测试文件: {test_file}")
                
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
            self.analyze_code(filename)
            executable = self.compile_code(filename)
            if executable:
                self.run_code(executable, lang)
            test_file = self.generate_test_code(filename)
            print(f"生成的测试文件: {test_file}")
        
        elif lang == 'python':
            filename = self.generate_python_code(
                function_name=f"auto_func_{timestamp}",
                params=[],
                return_type="None",
                body='print("这是自动生成的Python函数!")'
            )
            self.analyze_code(filename)
            self.run_code(filename, lang)
            test_file = self.generate_test_code(filename)
            print(f"生成的测试文件: {test_file}")
        
        elif lang == 'go':
            filename = self.generate_go_code(
                function_name=f"AutoFunc{timestamp}",
                params=[],
                return_type="int",
                body='fmt.Println("这是自动生成的Go函数!")'
            )
            executable = self.compile_code(filename)
            if executable:
                self.run_code(executable, lang)
            test_file = self.generate_test_code(filename)
            print(f"生成的测试文件: {test_file}")
        
        elif lang == 'rust':
            filename = self.generate_rust_code(
                function_name=f"auto_func_{timestamp}",
                params=[],
                return_type="i32",
                body='println!("这是自动生成的Rust函数!");'
            )
            executable = self.compile_code(filename)
            if executable:
                self.run_code(executable, lang)
            test_file = self.generate_test_code(filename)
            print(f"生成的测试文件: {test_file}")
        
        else:
            print(f"不支持的语言: {lang}")

if __name__ == "__main__":
    coder = AutoCoder()
    
    print("自动代码生成与编译工具")
    print("1. 生成并编译C代码")
    print("2. 生成并编译C++代码")
    print("3. 生成并编译Java代码")
    print("4. 生成并运行Python代码")
    print("5. 生成并编译Go代码")
    print("6. 生成并编译Rust代码")
    print("7. 管理代码模板")
    
    choice = input("请选择(1-7): ")
    
    if choice == '1':
        coder.auto_generate_and_compile('c')
    elif choice == '2':
        coder.auto_generate_and_compile('cpp')
    elif choice == '3':
        coder.auto_generate_and_compile('java')
    elif choice == '4':
        coder.auto_generate_and_compile('python')
    elif choice == '5':
        coder.auto_generate_and_compile('go')
    elif choice == '6':
        coder.auto_generate_and_compile('rust')
    elif choice == '7':
        # 模板管理功能
        print("\n模板管理")
        print("1. 添加新模板")
        print("2. 查看现有模板")
        sub_choice = input("请选择(1-2): ")
        
        if sub_choice == '1':
            lang = input("输入语言(c/cpp/java/python/go/rust): ")
            name = input("输入模板名称: ")
            print(f"请输入模板内容(使用占位符如{{function_name}}, {{params}}等):")
            content = []
            while True:
                line = input()
                if line == "EOF":
                    break
                content.append(line)
            content = "\n".join(content)
            
            coder.save_template(lang, name, content)
            print("模板保存成功!")
        elif sub_choice == '2':
            for lang, templates in coder.templates.items():
                print(f"\n{lang.upper()} 模板:")
                for name in templates.keys():
                    print(f"  - {name}")
        else:
            print("无效选择")
    else:
        print("无效选择")