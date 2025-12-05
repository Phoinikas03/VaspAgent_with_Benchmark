import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import openai

class IncarGenerator:
    def __init__(self, config_path, html_dir, task_desc, api_key, base_url, model_name, output_dir):
        self.config_path = config_path
        self.html_dir = html_dir
        self.task_desc = task_desc
        self.parameters = {}
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.output_dir = output_dir

    def load_config(self):
        """加载配置文件并过滤启用的HTML文件"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return [fname for fname, enabled in config.items() if enabled]

    @staticmethod
    def read_html(file_path):
        """使用BeautifulSoup提取关键内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                
                # 移除无关元素
                for tag in ['script', 'style', 'header', 'footer', 'nav']:
                    for element in soup(tag):
                        element.decompose()
                
                # 提取主要内容
                main_content = soup.find('main') or soup.find('article') or soup.body
                if not main_content:
                    return "No valid content found"
                
                # 结构化处理
                content = []
                for elem in main_content.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'pre']):
                    if elem.name.startswith('h'):
                        content.append(f"【{elem.text.strip()}】")
                    elif elem.name == 'ul':
                        items = [f"• {li.text.strip()}" for li in elem.find_all('li')]
                        content.append('\n'.join(items))
                    elif elem.name == 'pre':
                        content.append(f"代码块:\n{elem.text.strip()}")
                    else:
                        content.append(elem.text.strip())
                
                return '\n'.join(content)
        
        except Exception as e:
            print(f"解析失败 {file_path}: {str(e)}")
            return None

    def generate_prompt(self, param_name, html_content):
        """构造更严格的提示词"""
        return f"""作为VASP专家，请严格按照要求为[{self.task_desc}]任务设置{param_name}参数：

        【参数文档】
        {html_content}

        【输出要求】
        1. 参数值必须严格符合VASP语法，且只包含数值部分
        2. 绝对不要包含参数名称和等号(=)
        3. 数值格式示例：
        - 布尔值：.TRUE. 或 .FALSE.
        - 数值：520 或 1e-5
        - 多个值：4.0 4.0 0.0

        【输出格式】
        参数值： <纯数值>
        理由： <50字说明>

        示例：
        参数值： .TRUE.
        理由： 需要开启自洽计算功能"""

    @staticmethod
    def parse_response(response):
        """增强的解析逻辑"""
        # 严格匹配数值模式
        pattern = r"参数值[：:]\s*((?:\.?[A-Z]+\.?|\d+[\d\.eE\- ]*)+)"
        match = re.search(pattern, response, re.IGNORECASE)
        
        if match:
            value = match.group(1).strip()
            # 清理可能的残留符号
            value = re.sub(r"[^\.\dA-Za-z\-eE ]", "", value)
            # 统一布尔值格式
            if re.match(r"^(T|TRUE|\.TRUE\.)$", value, re.IGNORECASE):
                return ".TRUE."
            if re.match(r"^(F|FALSE|\.FALSE\.)$", value, re.IGNORECASE):
                return ".FALSE."
            return value
        
        # 备选解析方案
        backup_pattern = r"(?:=|：)\s*([\.\dA-Za-z\-eE ]+?)(?=\s|$)"
        backup_match = re.search(backup_pattern, response)
        return backup_match.group(1).strip() if backup_match else None

    def process_single_file(self, filename):
        """处理单个参数文件"""
        param_name = os.path.splitext(filename)[0].upper()  # 确保参数名大写
        file_path = os.path.join(self.html_dir, filename)
        
        html_content = self.read_html(file_path)
        if not html_content:
            raise FileNotFoundError(f"无法读取文件: {file_path}")

        prompt = self.generate_prompt(param_name, html_content)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256
            )
            llm_output = response.choices[0].message.content.strip()
            param_value = self.parse_response(llm_output)
            
            if param_value:
                self.parameters[param_name] = param_value
                print(f"成功解析 {param_name}={param_value}")
            else:
                print(f"无法解析 {param_name}: {llm_output}")
            
        except Exception as e:
            print(f"API调用失败 {param_name}: {str(e)}")

    def generate_incar(self, filename="INCAR"):
        """生成VASP输入文件"""
        # 预设推荐参数（可调整）
        recommended_params = {
            "PREC": "Normal",
            "ALGO": "Fast",
            "ISMEAR": "0",
            "SIGMA": "0.05",
            "LREAL": "Auto",
            "LWAVE": ".FALSE.",
            "LCHARG": ".FALSE."
        }
        
        # 合并LLM生成的参数
        final_params = {**recommended_params, **self.parameters} # 后面的优先级更高
        
        # 按VASP推荐顺序排序
        priority_order = [
            "SYSTEM", "PREC", "ENCUT", "EDIFF", "EDIFFG", 
            "IBRION", "ISIF", "NSW", "POTIM", 
            "ISMEAR", "SIGMA", "LORBIT", 
            "ALGO", "LREAL", "LWAVE", "LCHARG"
        ]
        
        with open(filename, 'w', encoding='utf-8') as f:
            # 先写入优先级参数
            for param in priority_order:
                if param in final_params:
                    f.write(f"{param} = {final_params[param]}\n")
                    del final_params[param]
            
            # 写入剩余参数
            for param, value in final_params.items():
                f.write(f"{param} = {value}\n")
        
        print(f"成功生成 {filename}")

    def run_parallel(self, max_workers=4):
        """并行处理参数"""
        target_files = self.load_config()
        print(f"开始处理 {len(target_files)} 个参数...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_single_file, target_files)
        
        self.generate_incar(f'{self.output_dir}/INCAR')

if __name__ == "__main__":
    # 配置参数
    CONFIG = {
        "config_path": "/home/xiazeyu21/Programs/vasp_agent/INCAR_tags/config1.json",
        "html_dir": "/home/xiazeyu21/Programs/vasp_agent/INCAR_tags/html",
        "task_desc": "生成结构优化INCAR",
        "api_key": "sk-d69124571de44156ab4b8fba9f289fc7",
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-reasoner",
        "output_dir": "/home/xiazeyu21/Programs/vasp_agent/INCAR_tags"
    }

    # 执行生成流程
    generator = IncarGenerator(**CONFIG)
    generator.run_parallel(max_workers=16)