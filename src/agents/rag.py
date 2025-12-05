import logging
from typing import List, Dict, Any
import os, re, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
# from src.llm import ChatEngine
import pickle
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from src.agents.base import agent_manager

logger = logging.getLogger(__name__)

@agent_manager.register("rag")
class RAGAgent:
    def __init__(
        self, 
        model: str,
        docs_dir: str = "database/INCAR_tags/rag_ready",
        embedding_model: str = "models/all-MiniLM-L6-v2",
        embedding_file: str = "embeddings.pkl",
        top_k: int = 3,
        max_attempt: int = 3,
        **kwargs
    ):
        logger.info(f"Initializing VASPRAGAgent with model: {model}")
        # Initialize the LLM engine
        self.llm = ChatEngine(model, **kwargs)
        self.top_k = top_k
        self.max_attempt = max_attempt
        
        # Load documentation
        logger.info(f"Loading documents from: {docs_dir}")
        self.docs_dir = Path(docs_dir)
        self.docs = self._load_docs()
        logger.info(f"Loaded {len(self.docs)} documents")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embed_model = SentenceTransformer(embedding_model)
        
        # Set embeddings file path
        self.embedding_file = Path(embedding_file)
        
        # Try to load existing embeddings or create new ones
        self.doc_embeddings = self._load_or_create_embeddings()
        logger.info("Document embeddings loaded/created successfully")

    def _save_embeddings(self):
        """Save embeddings to file."""
        try:
            with open(self.embedding_file, 'wb') as f:
                pickle.dump(self.doc_embeddings, f)
            logger.info(f"Embeddings saved to {self.embedding_file}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load embeddings from file."""
        try:
            with open(self.embedding_file, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"Embeddings loaded from {self.embedding_file}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            return None

    def _load_or_create_embeddings(self) -> Dict[str, np.ndarray]:
        """Load existing embeddings if available, otherwise create new ones."""
        if self.embedding_file.exists():
            logger.info("Loading embeddings...")
            embeddings = self._load_embeddings()
            # Verify if loaded embeddings match current documents
            if embeddings and set(embeddings.keys()) == set(self.docs.keys()):
                return embeddings
            logger.info("Existing embeddings don't match current documents")
        
        logger.info("Creating new embeddings...")
        embeddings = self._create_embeddings()
        self._save_embeddings()
        return embeddings

    def _load_docs(self) -> Dict[str, str]:
        """Load all documentation files."""
        docs = {}
        for file_path in self.docs_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    tag_name = file_path.stem
                    docs[tag_name] = content
                    logger.debug(f"Loaded document: {tag_name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        return docs

    def _create_embeddings(self) -> Dict[str, np.ndarray]:
        """Create embeddings for all documents."""
        embeddings = {}
        total_docs = len(self.docs)
        for idx, (tag, content) in enumerate(self.docs.items(), 1):
            try:
                embedding = self.embed_model.encode(content)
                embeddings[tag] = embedding
                logger.debug(f"Created embedding for {tag} ({idx}/{total_docs})")
            except Exception as e:
                logger.error(f"Error creating embedding for {tag}: {str(e)}")
        return embeddings

    def _get_relevant_docs(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve most relevant documents for the query."""
        # Get query embedding
        query_embedding = self.embed_model.encode(query)
        
        # Calculate similarities
        similarities = {}
        for tag, doc_embedding in self.doc_embeddings.items():
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities[tag] = similarity
        
        # Get top-k most similar docs
        top_tags = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        relevant_docs = []
        for tag, score in top_tags:
            relevant_docs.append({
                "tag": tag,
                "content": self.docs[tag],
                "score": float(score)
            })
        
        return relevant_docs
    
    def query(self, question: str, top_k: int = None) -> str:
        """Query the RAG system with a question."""
        if top_k is None: top_k = self.top_k
        # Get relevant documents
        relevant_docs = self._get_relevant_docs(question, top_k)
        
        # Construct prompt with context
        context = "\n\n".join([
            f"Tag: {doc['tag']}\nContent: {doc['content']}" 
            for doc in relevant_docs
        ])

        logger.info(f"Constructed prompt with context: {[doc['tag'] for doc in relevant_docs]}")
        
        prompt = f"""Based on the following documentation, please answer the question.

Context:
{context}

Question: {question}

Please provide a clear and concise answer based on the provided documentation. If the information is not available in the context, please state that."""
        
        # Get response from LLM
        response = self.llm.chat(prompt)
        return response

    def chat(self, message: str, top_k: int = None, max_attempt: int = None) -> str:
        """Chat with the RAG system."""
        if max_attempt is None: max_attempt = self.max_attempt
        pattern = r"```query(.*?)```"
        attempt = 0
        original_message = message

        response = self.llm.chat(message)
        match = re.search(pattern, response, re.DOTALL)
        if match: 
            query = match.group(1).strip()
            while attempt < max_attempt:
                attempt += 1
                logger.info(f"Attempt {attempt} for query: {query}")
                response = self.query(query, top_k)
                match = re.search(pattern, response, re.DOTALL)
                if match: query = match.group(1).strip()
                else: break
            if attempt >= max_attempt:
                logger.warning(f"Max attempt reached for query: {original_message}")
                response = self.llm.chat(f"You have reached the maximum number of attempts. Please directly respond to the question:\n {original_message}")

        return response

    def reset(self):
        """Reset the conversation history."""
        self.llm.reset()

@agent_manager.register("incar")
class INCARAgent:
    def __init__(self, 
        model: str,
        config_path, 
        html_dir, 
        **kwargs
    ):
        self.config_path = config_path
        self.html_dir = html_dir
        self.parameters = {}
        self.llm = ChatEngine(model, **kwargs)

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
        return f"""作为VASP专家，请严格按照要求为[{self.query}]任务设置{param_name}参数：

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
            response = self.llm.create([{"role": "user", "content": prompt}])
            param_value = self.parse_response(response)
            
            if param_value:
                self.parameters[param_name] = param_value
                print(f"成功解析 {param_name}={param_value}")
            else:
                print(f"无法解析 {param_name}: {response}")
            
        except Exception as e:
            print(f"API调用失败 {param_name}: {str(e)}")

    def generate_incar(self):
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
        
        content = ""
        # 先写入优先级参数
        for param in priority_order:
            if param in final_params:
                content += f"{param} = {final_params[param]}\n"
                del final_params[param]
        
        # 写入剩余参数
        for param, value in final_params.items():
            content += f"{param} = {value}\n"

        print("成功生成INCAR")

        return content

    def run_parallel(self, query, max_workers=4):
        self.query = query
        """并行处理参数"""
        target_files = self.load_config()
        print(f"开始处理 {len(target_files)} 个参数...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_single_file, target_files)
        
        content = self.generate_incar()

        return content

    def chat(self, message, max_workers=4):
        return self.run_parallel(message, max_workers)

    def reset(self):
        self.parameters = {}
        self.llm.reset()