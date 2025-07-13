import os
import subprocess
import sys
import time
from datetime import datetime
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import platform
import argparse
from dataclasses import dataclass

# 常量定义
SUPPORTED_LANGUAGES = ["c", "cpp", "java", "python", "go", "rust"]
DEFAULT_TEMPLATES = {
    lang: {"default": f"default_{lang}_template"} for lang in SUPPORTED_LANGUAGES
}

@dataclass
class CodeMethod:
    name: str
    params: List[str]
    return_type: str
    body: str

class CodeGenerator:
    def __init__(self):
        self.template_dir = Path("templates")
        self.output_dir = Path("generated_code")
        self.template_db = self.template_dir / "template_db.json"
        self.setup_environment()
        
    def setup_environment(self):
        """初始化工作目录和模板数据库"""
        self.template_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        if not self.template_db.exists():
            self._init_template_db()

    def _init_template_db(self):
        """初始化模板数据库"""
        with open(self.template_db, 'w') as f:
            json.dump(DEFAULT_TEMPLATES, f, indent=2)

    def _get_compiler_path(self, lang: str) -> Optional[str]:
        """获取编译器路径"""
        compilers = {
            "c": "gcc",
            "cpp": "g++",
            "java": "javac",
            "go": "go",
            "rust": "rustc"
        }
        compiler = compilers.get(lang)
        if compiler and shutil.which(compiler):
            return compiler
        return None

    def generate_code(self, lang: str, **kwargs) -> Path:
        """
        生成代码的通用接口
        """
        generator = getattr(self, f"_generate_{lang}_code", None)
        if not generator:
            raise ValueError(f"Unsupported language: {lang}")
            
        filename = generator(**kwargs)
        self._format_code(filename, lang)
        return filename

    def _generate_c_code(self, function_name: str, params: List[str], 
                        return_type: str, body: str) -> Path:
        """生成C代码"""
        code = f"""#include <stdio.h>

{return_type} {function_name}({', '.join(params)}) {{
    {body}
    return 0;
}}

int main() {{
    {function_name}();
    printf("程序执行成功!\\n");
    return 0;
}}
"""
        filename = self.output_dir / f"{function_name}.c"
        filename.write_text(code)
        return filename

    # 其他语言的生成函数类似，这里省略...
    
    def _format_code(self, filename: Path, lang: str):
        """代码格式化"""
        formatters = {
            "python": "black",
            "c": "clang-format",
            "cpp": "clang-format",
            "java": "google-java-format",
            "go": "gofmt",
            "rust": "rustfmt"
        }
        
        if formatter := formatters.get(lang):
            try:
                subprocess.run([formatter, str(filename)], check=True)
            except subprocess.CalledProcessError:
                print(f"警告: {formatter} 格式化失败")

    def compile_code(self, filename: Path, lang: str) -> Optional[Path]:
        """编译代码"""
        if not (compiler := self._get_compiler_path(lang)):
            print(f"错误: 未找到{lang}编译器")
            return None

        output_name = filename.stem
        try:
            if lang == "c":
                cmd = [compiler, str(filename), "-o", str(self.output_dir/output_name)]
            elif lang == "cpp":
                cmd = [compiler, str(filename), "-o", str(self.output_dir/output_name)]
            elif lang == "java":
                cmd = [compiler, str(filename)]
            elif lang == "go":
                cmd = [compiler, "build", "-o", str(self.output_dir/output_name), str(filename)]
            elif lang == "rust":
                cmd = [compiler, str(filename), "-o", str(self.output_dir/output_name)]

            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return self.output_dir / output_name
        except subprocess.CalledProcessError as e:
            print(f"编译失败: {e.stderr}")
            return None

    def run_code(self, executable: Path, lang: str):
        """运行代码"""
        try:
            if lang == "java":
                # Java需要在输出目录运行
                cwd = self.output_dir
                cmd = ["java", executable.stem]
            else:
                cwd = None
                cmd = [str(executable)]

            result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"运行时警告: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"运行失败: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(description="高级代码生成工具")
    parser.add_argument("language", choices=SUPPORTED_LANGUAGES, help="要生成的语言")
    parser.add_argument("--name", default=f"auto_{datetime.now().strftime('%Y%m%d%H%M%S')}", 
                       help="生成的函数/类名")
    args = parser.parse_args()

    generator = CodeGenerator()
    
    # 根据语言生成不同的代码结构
    if args.language == "python":
        code_file = generator.generate_code(
            lang=args.language,
            function_name=args.name,
            params=[],
            return_type="None",
            body='print("这是自动生成的Python函数!")'
        )
    elif args.language == "cpp":
        code_file = generator.generate_code(
            lang=args.language,
            class_name=args.name,
            methods=[CodeMethod(
                name="printMessage",
                params=[],
                return_type="void",
                body='std::cout << "这是自动生成的C++方法!" << std::endl;'
            )]
        )
    # 其他语言类似...
    
    # 编译和运行
    if executable := generator.compile_code(code_file, args.language):
        generator.run_code(executable, args.language)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)
