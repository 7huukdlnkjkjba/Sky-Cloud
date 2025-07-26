import os
import subprocess
import sys
import time
from datetime import datetime
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import platform
import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
import inspect
from jinja2 import Environment, FileSystemLoader

class Language(Enum):
    C = auto()
    CPP = auto()
    JAVA = auto()
    PYTHON = auto()
    GO = auto()
    RUST = auto()

@dataclass
class CodeMethod:
    name: str
    params: List[Tuple[str, str]] = field(default_factory=list)  # (type, name)
    return_type: str = "void"
    body: str = ""
    docstring: str = ""

@dataclass
class CodeTemplate:
    name: str
    content: str
    language: Language
    metadata: Dict[str, Any] = field(default_factory=dict)

class CodeGenerator:
    def __init__(self, verbose: bool = False):
        self.template_dir = Path("templates")
        self.output_dir = Path("generated_code")
        self.template_db = self.template_dir / "template_db.json"
        self.verbose = verbose
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.setup_environment()

    def log(self, message: str, level: str = "INFO"):
        if self.verbose or level == "ERROR":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def setup_environment(self):
        """Initialize working directories and template database"""
        self.template_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        if not self.template_db.exists():
            self._init_template_db()
        
        self._load_template_system()

    def _init_template_db(self):
        """Initialize template database with default templates"""
        default_templates = {
            lang.name.lower(): {
                "default": f"default_{lang.name.lower()}_template",
                "templates": {}
            } for lang in Language
        }
        with open(self.template_db, 'w') as f:
            json.dump(default_templates, f, indent=2)
        self.log("Initialized new template database")

    def _load_template_system(self):
        """Load and validate template system"""
        try:
            with open(self.template_db) as f:
                self.templates = json.load(f)
            
            # Verify template files exist
            for lang, data in self.templates.items():
                for tpl_name, tpl_file in data["templates"].items():
                    tpl_path = self.template_dir / f"{tpl_file}.j2"
                    if not tpl_path.exists():
                        self.log(f"Missing template file: {tpl_path}", "WARNING")
        except Exception as e:
            self.log(f"Failed to load template system: {str(e)}", "ERROR")
            raise

    def _get_compiler_path(self, lang: str) -> Optional[str]:
        """Get compiler path with version check"""
        compilers = {
            "c": "gcc",
            "cpp": "g++",
            "java": "javac",
            "go": "go",
            "rust": "rustc"
        }
        
        compiler = compilers.get(lang)
        if not compiler:
            return None

        if not shutil.which(compiler):
            self.log(f"Compiler not found: {compiler}", "WARNING")
            return None

        try:
            # Verify compiler version
            result = subprocess.run(
                [compiler, "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.log(f"Using {compiler}: {result.stdout.splitlines()[0]}")
                return compiler
        except Exception as e:
            self.log(f"Compiler check failed: {str(e)}", "WARNING")
        
        return None

    def generate_code(self, lang: str, template: str = "default", **kwargs) -> Path:
        """
        Generate code using specified template with validation
        """
        lang = lang.lower()
        if lang not in self.templates:
            raise ValueError(f"Unsupported language: {lang}")
        
        if template not in self.templates[lang]["templates"]:
            self.log(f"Template '{template}' not found, using default", "WARNING")
            template = "default"

        try:
            # Try Jinja2 template first
            tpl_file = self.templates[lang]["templates"][template]
            return self._generate_from_template(lang, tpl_file, **kwargs)
        except Exception as e:
            self.log(f"Template rendering failed: {str(e)}, falling back to hardcoded", "WARNING")
            generator = getattr(self, f"_generate_{lang}_code", None)
            if not generator:
                raise ValueError(f"No generator for language: {lang}")
            return generator(**kwargs)

    def _generate_from_template(self, lang: str, template: str, **kwargs) -> Path:
        """Generate code using Jinja2 template"""
        template = self.jinja_env.get_template(f"{template}.j2")
        rendered = template.render(**kwargs)
        
        filename = self.output_dir / f"{kwargs.get('name', 'output')}.{lang}"
        filename.write_text(rendered)
        self.log(f"Generated {filename} from template")
        return filename

    def _generate_python_code(self, name: str, methods: List[CodeMethod], **kwargs) -> Path:
        """Generate Python class with methods"""
        imports = kwargs.get("imports", ["typing"])
        
        code = f"# Auto-generated Python code\n"
        code += "\n".join(f"import {imp}" for imp in imports) + "\n\n"
        
        if kwargs.get("use_class", True):
            code += f"class {name}:\n"
            indent = "    "
        else:
            code += f"# Module-level functions\n"
            indent = ""
        
        for method in methods:
            params = ", ".join([f"{name}: {type}" for type, name in method.params])
            code += f"{indent}def {method.name}({params}) -> {method.return_type}:\n"
            code += f'{indent}    """{method.docstring}"""\n' if method.docstring else ""
            code += f"{indent}    {method.body}\n\n"
        
        if kwargs.get("add_main", True):
            code += "\nif __name__ == '__main__':\n"
            if kwargs.get("use_class", True):
                code += f"    obj = {name}()\n"
                code += f"    obj.{methods[0].name}()\n"
            else:
                code += f"    {methods[0].name}()\n"
            code += '    print("Execution successful!")\n'
        
        filename = self.output_dir / f"{name}.py"
        filename.write_text(code)
        return filename

    # ... (其他语言的生成方法类似，保持原有实现)

    def compile_and_run(self, filename: Path, lang: str, args: List[str] = None) -> bool:
        """End-to-end compilation and execution pipeline"""
        args = args or []
        
        if not filename.exists():
            self.log(f"File not found: {filename}", "ERROR")
            return False

        # Step 1: Compile (if needed)
        executable = self.compile_code(filename, lang)
        if not executable and lang not in ["python"]:
            return False

        # Step 2: Run
        self.run_code(executable or filename, lang, args)
        return True

    # ... (保持原有compile_code和run_code实现，增加args参数传递)

def main():
    parser = argparse.ArgumentParser(description="Advanced Code Generation Tool")
    parser.add_argument("language", 
                       choices=[lang.name.lower() for lang in Language],
                       help="Target programming language")
    parser.add_argument("--name", 
                       default=f"auto_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                       help="Generated class/function name")
    parser.add_argument("--template", default="default",
                       help="Template to use for generation")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                       help="Arguments to pass to generated program")
    
    args = parser.parse_args()

    generator = CodeGenerator(verbose=args.verbose)
    
    try:
        # 示例方法生成
        methods = [
            CodeMethod(
                name="main",
                params=[],
                return_type="None",
                body='print("Hello from generated code!")',
                docstring="Main entry point"
            )
        ]
        
        code_file = generator.generate_code(
            lang=args.language,
            template=args.template,
            name=args.name,
            methods=methods,
            use_class=args.language not in ["python", "go"]  # 部分语言更适合面向对象
        )
        
        success = generator.compile_and_run(
            code_file,
            args.language,
            args.args
        )
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        generator.log(f"Generation failed: {str(e)}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
