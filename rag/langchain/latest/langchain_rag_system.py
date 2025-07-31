#!/usr/bin/env python3
"""
基于LangChain的增强版Topic-RAG系统
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangChain导入
from langchain_core.language_models import LLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import models
# 添加根目录到Python路径
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

sys.path.append(os.path.join(root_dir, 'src'))
sys.path.append(os.path.join(root_dir, 'rag', 'utils'))

try:
    from src.topic_model import MultiOmicsETM
    from load_separate_models import load_separate_models
    print("✅ Successfully imported load_separate_models from root directory")

    try:
        from deepseek_translate import deepseek_translate
        print("✅ Successfully imported deepseek_translate")
    except ImportError:
        try:
            # 尝试从rag.utils导入
            from rag.utils.deepseek_translate import deepseek_translate
            print("✅ Successfully imported deepseek_translate from rag.utils")
        except ImportError:
            print("⚠️ deepseek_translate not available, using fallback")
            def deepseek_translate(text, target_lang="en", api_key=None):
                return text

except ImportError as e:
    print(f"❌ Model import failed: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")

    # Create a simple placeholder function
    def load_separate_models(model_dir):
        print("Warning: Using placeholder model loading function")
        return None, None, None, None
    def deepseek_translate(text, target_lang="en", api_key=None):
        print("Warning: Translation function not available")
        return text

class DeepSeekLangChainLLM(LLM):
    """DeepSeek LLM的LangChain包装器"""
    
    api_key: str
    model_name: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/v1"
    
    def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
        super().__init__(api_key=api_key, model_name=model_name)
        
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """调用DeepSeek API"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"API调用失败: {response.status_code}"
                
        except Exception as e:
            return f"调用失败: {str(e)}"

import base64
import io
from PIL import Image
from openai import OpenAI

class LangChainTopicRAGSystem:
    """基于LangChain的完整Topic-RAG系统"""
    
    def __init__(self, model_dir='/root/autodl-tmp/AETM/models', device='cpu', api_key=None, doubao_api_key=None):
        """
        Initialize complete LangChain Topic-RAG system
        """
        print(f"🚀 Initializing LangChain Topic-RAG System...")
        print(f"   Model directory: {model_dir}")
        print(f"   Device: {device}")

        self.device = device
        self.model_dir = model_dir
        self.api_key = api_key
        self.doubao_api_key = doubao_api_key

        # Load models and components
        print("📊 Loading complete models and components...")
        try:
            self.model, self.vectorizer, self.vocab, self.theta = load_separate_models(model_dir)
            print("✅ Model components loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

        if self.model is None:
            raise Exception("Model loading failed, please check if model files exist")

        self.model.to(device)
        self.model.eval()

        # Load original data - using absolute path
        data_path = '/root/autodl-tmp/AETM/data/palettes_descriptions.xlsx'
        print(f"📚 Loading data file: {data_path}")

        if not os.path.exists(data_path):
            print(f"❌ Data file does not exist: {data_path}")
            # Try to list directory contents
            data_dir = os.path.dirname(data_path)
            if os.path.exists(data_dir):
                print(f"Data directory contents: {os.listdir(data_dir)}")
            raise FileNotFoundError(f"Data file does not exist: {data_path}")

        try:
            self.df_plans = pd.read_excel(data_path)
            print(f"✅ Data file loaded successfully, {len(self.df_plans)} records")
        except Exception as e:
            print(f"❌ Data file reading failed: {e}")
            raise

        # Initialize LangChain components
        print("🔧 Initializing LangChain components...")
        self._init_langchain_components()

        # Build complete retrieval database
        print("🗃️ Building complete retrieval database...")
        self._build_complete_retrieval_database()

        print("✅ Complete LangChain Topic-RAG system initialization completed!")
    
    def _translate_text(self, text: str, target_lang: str = "en") -> str:
        """Translate text using DeepSeek API"""
        try:
            if not self.api_key:
                print("Warning: No API key provided, translation skipped")
                return text

            translated = deepseek_translate(text, target_lang=target_lang, api_key=self.api_key)
            return translated
        except Exception as e:
            print(f"Translation failed: {e}")
            return text

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Count Chinese characters
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        total_chars = len(text.replace(' ', ''))

        if total_chars == 0:
            return "en"

        chinese_ratio = chinese_chars / total_chars
        return "zh" if chinese_ratio > 0.3 else "en"

    def _get_fewshot_examples(self):
        """获取Few-Shot范例，用于教会LLM按指定数量生成颜色"""
        examples = {
            "3_colors": {
                "input": "I need an extremely minimalist color scheme for a coffee brand logo, only 3 colors needed.",
                "output": """### **Minimalist Coffee Logo**
**Design Concept**: Convey warmth and professionalism through ultimate simplicity. Inspired by a fresh latte, using minimal colors to express core concepts: coffee, milk, and focus.
---
### **Color Scheme**
1. **Espresso Black - RGB(50, 40, 35)**
   *Represents the coffee base, stable and professional.*
2. **Warm Milk White - RGB(245, 240, 235)**
   *Represents blended milk, warm and pure.*
3. **Terracotta Brown - RGB(180, 110, 80)**
   *Represents the coffee cup, adding handcrafted warmth and texture.*"""
            },
            "7_colors": {
                "input": "Design a 7-color environmental palette for a fantasy RPG game set in a magical forest.",
                "output": """### **Magical Forest "Elven Twilight"**
**Design Concept**: Capture moonlight filtering through ancient forest canopy, illuminating magical creatures and plants. Create an atmosphere that's both serene and filled with potential danger and wonder.
---
### **Color Scheme**
1. **Midnight Blue - RGB(20, 25, 55)**
   *Forest's night backdrop, vast and mysterious.*
2. **Ancient Wood Brown - RGB(60, 45, 40)**
   *Massive tree trunks, full of power.*
3. **Moss Green - RGB(80, 110, 70)**
   *Moss covering roots and stones, representing life and moisture.*
4. **Moonlight Silver - RGB(210, 220, 230)**
   *Cold moonlight piercing through leaves, main light source.*
5. **Ghost Mushroom Cyan - RGB(100, 200, 180)**
   *Glowing magical mushrooms, fantasy elements and accent lights.*
6. **Fairy Blood Red - RGB(190, 50, 60)**
   *Hidden magical flowers or warning colors of dangerous creatures.*
7. **Earth Gray - RGB(100, 95, 90)**
   *Forest rocks and soil, providing neutral transitions.*"""
            }
        }
        return examples

    def _create_fewshot_prompt(self, num_colors: int, user_query: str, retrieved_context: str = "") -> str:
        """使用Few-Shot方法创建提示词，确保生成指定数量的颜色"""
        examples = self._get_fewshot_examples()

        # 选择合适的范例
        if num_colors <= 4:
            primary_example = examples["3_colors"]
            secondary_example = examples["7_colors"]
        else:
            primary_example = examples["7_colors"]
            secondary_example = examples["3_colors"]

        prompt = f"""# Role
You are a top-tier color design master who can precisely generate the specified number of colors according to user requirements and provide complete design solutions.

# Task
Please learn from the following examples to understand how to construct your answer based on the specified number of colors. Then, create a brand new color scheme based on the final user's real requirements.

---
[Example 1: Input]
User Requirements: "{primary_example['input']}"
Reference Cases: (omitted or empty)

[Example 1: Output]
{primary_example['output']}

---
[Example 2: Input]
User Requirements: "{secondary_example['input']}"
Reference Cases: (omitted or empty)

[Example 2: Output]
{secondary_example['output']}

---

# User's Real Requirements and Reference Cases
Now, please strictly follow the format of the above examples and generate a brand new color scheme containing **{num_colors}** colors based on the following user's real requirements.

[Real Requirements: Input]
User Requirements: "{user_query}"
Reference Cases: "{retrieved_context}"

[Real Requirements: Output]
"""

        return prompt

    def _init_langchain_components(self):
        """Initialize LangChain components"""
        if not self.api_key:
            print("Warning: No API key provided, LangChain functionality will be limited")
            # Set default values to avoid attribute errors
            self.llm = None
            self.memory = None
            self.rag_generation_chain = None
            return

        # Initialize LLM
        self.llm = DeepSeekLangChainLLM(self.api_key)

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create Prompt templates
        self._create_prompt_templates()

        # Create processing chains
        self._create_processing_chains()

        print("✅ LangChain components initialized successfully")
    
    def _create_prompt_templates(self):
        """Create Prompt templates"""

        # Image understanding template
        self.image_understanding_template = PromptTemplate(
            input_variables=["image_path"],
            template="""Please analyze the design features of image {image_path}, including:
1. Overall design style (modern, classical, minimalist, luxurious, etc.)
2. Main color combinations and tones
3. Design elements and layout characteristics
4. Applicable scenarios and space types
5. Emotional atmosphere created

Please describe in concise and professional language, focusing on color combinations."""
        )

        # Text fusion template
        self.text_fusion_template = PromptTemplate(
            input_variables=["user_text", "image_analysis"],
            template="""Please generate a detailed design scheme description based on the following information:

User Requirements: {user_text}
Image Analysis: {image_analysis}

Please integrate user requirements and image analysis to generate a detailed design scheme description, including:
1. Overall design style (combining user requirements and image characteristics)
2. Color matching suggestions (based on image colors and user preferences)
3. Design element characteristics (integrating image elements and user requirements)
4. Applicable scenarios (clearly specify the usage environment)
5. Emotional atmosphere (describe the specific emotions created by the design)

Requirements: Professional language, specific descriptions, highlighting color combinations, reflecting personalization."""
        )

        # RAG generation template
        self.rag_generation_template = PromptTemplate(
            input_variables=["prompt"],
            template="""You are a professional design aesthetics expert. Please generate a brand new, personalized design scheme based on the following information:

{prompt}

Please use the above reference schemes as inspiration, combined with user requirements, to generate a completely new design scheme. Requirements:

1. **Design Style**: Create a new style different from the reference schemes, clearly explain the innovation points
2. **Color Matching**: Generate 5 brand new RGB color values that are significantly different from the reference scheme colors
3. **Design Elements**: Describe innovative design elements and layouts, reflecting originality
4. **Applicable Scenarios**: Expand or change the application scenarios of the reference schemes
5. **Emotional Atmosphere**: Create an emotional atmosphere different from the reference schemes

**Important Requirements**:
- The generated RGB color values must be significantly different from the colors in the reference schemes
- The design style should be different from the reference schemes
- Ensure it is a brand new creative work, not a simple modification of the reference schemes
- Color combinations should have logic and aesthetic value

Please answer in English with professional and elegant language, highlighting innovation and personalization."""
        )
    
    def _create_processing_chains(self):
        """创建处理链"""
        
        # 简化为单个RAG生成链
        self.rag_generation_chain = LLMChain(
            llm=self.llm,
            prompt=self.rag_generation_template,
            memory=self.memory
        )
    
    def _build_complete_retrieval_database(self):
        """构建完整的检索数据库（包含文本和颜色向量）"""
        print("正在构建完整检索数据库...")
        
        # 使用训练好的theta矩阵作为主题表示
        self.theta_vectors = self.theta.cpu().numpy()
        
        # 构建颜色向量
        color_cols = [f'Color_{i}_{c}' for i in range(1, 6) for c in ['R', 'G', 'B']]
        self.color_vectors = self.df_plans[color_cols].values
        
        # 存储原始文本描述
        self.text_descriptions = self.df_plans['description'].values
        
        # 确保数组长度一致
        min_length = min(len(self.theta_vectors), len(self.color_vectors), len(self.text_descriptions))
        
        # 截取到最小长度
        self.theta_vectors = self.theta_vectors[:min_length]
        self.color_vectors = self.color_vectors[:min_length]
        self.text_descriptions = self.text_descriptions[:min_length]
        
        # 预计算归一化向量用于快速检索
        from sklearn.preprocessing import normalize
        self.theta_vectors_normalized = normalize(self.theta_vectors, norm='l2', axis=1)
        self.color_vectors_normalized = normalize(self.color_vectors, norm='l2', axis=1)
        
        print(f"完整检索数据库构建完成：{len(self.text_descriptions)}个文档")
    
    def _complete_topic_model_inference(self, fused_text: str) -> tuple:
        """完整的Topic Model推理（模块一）"""
        try:
            # 文本向量化
            text_bow = self._text_to_bow(fused_text)
            
            # Topic Model推理
            self.model.eval()
            with torch.no_grad():
                # 文本编码
                text_tensor = torch.FloatTensor(text_bow).to(self.device)
                mu_text, logvar_text = self.model.encode_text(text_tensor)
                
                # 颜色编码（使用零向量作为占位符）
                mu_color = torch.zeros_like(mu_text)
                logvar_color = torch.ones_like(logvar_text) * 10
                
                # 多模态融合
                mu, logvar = self.model.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
                theta = self.model.get_theta(mu)
                
                # 解码重构
                recon_color, recon_text = self.model.decode(theta)
                
                return (
                    recon_text.cpu().numpy(),
                    recon_color.cpu().numpy(),
                    theta.cpu().numpy(),
                    mu.cpu().numpy(),
                    logvar.cpu().numpy()
                )
                
        except Exception as e:
            print(f"Topic Model推理失败: {e}")
            return None, None, None, None, None

    def _dual_stage_retrieval(self, theta_query: np.ndarray, recon_color: np.ndarray, top_k: int = 10) -> List[Dict]:
        """双阶段检索：文本相似度 + 颜色相似度"""
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import normalize
        
        try:
            # 第一阶段：基于主题向量的文本相似度检索
            theta_query_norm = normalize(theta_query, norm='l2', axis=1)
            text_similarities = cosine_similarity(theta_query_norm, self.theta_vectors_normalized).flatten()
            
            # 获取文本Top-K候选（扩大候选池）
            text_top_k = min(top_k * 3, len(text_similarities))
            text_top_indices = np.argsort(text_similarities)[-text_top_k:][::-1]
            
            # 第二阶段：在候选中进行颜色相似度重排序
            candidate_results = []
            recon_color_norm = normalize(recon_color, norm='l2', axis=1)
            
            for idx in text_top_indices:
                # 计算颜色相似度
                candidate_color_vec = self.color_vectors_normalized[idx:idx+1]
                color_similarity = cosine_similarity(recon_color_norm, candidate_color_vec)[0][0]
                
                # 综合得分（可调整权重）
                text_score = text_similarities[idx]
                color_score = color_similarity
                combined_score = 0.6 * text_score + 0.4 * color_score
                
                candidate_results.append({
                    'index': idx,
                    'text_score': text_score,
                    'color_score': color_score,
                    'combined_score': combined_score,
                    'description': self.text_descriptions[idx],
                    'colors': self.color_vectors[idx].reshape(-1, 3)
                })
            
            # 按综合得分排序，返回Top-K
            candidate_results.sort(key=lambda x: x['combined_score'], reverse=True)
            return candidate_results[:top_k]
            
        except Exception as e:
            print(f"双阶段检索失败: {e}")
            return []

    def _text_to_bow(self, text: str) -> np.ndarray:
        """将文本转换为词袋向量"""
        try:
            if self.vectorizer is None:
                raise Exception("TF-IDF向量化器未加载")
            
            # 使用训练好的TF-IDF向量化器
            bow_vector = self.vectorizer.transform([text]).toarray().astype(np.float32)
            return bow_vector
            
        except Exception as e:
            print(f"文本向量化失败: {e}")
            # 返回零向量作为fallback
            vocab_size = len(self.vocab) if self.vocab else 2000
            return np.zeros((1, vocab_size), dtype=np.float32)

    def _retrieve_candidates_langchain(self, user_text: str, top_k: int = 10) -> List[Dict]:
        """使用LangChain方法检索候选方案"""
        try:
            # 使用完整的Topic Model推理
            recon_text, recon_color, theta_query, mu, logvar = self._complete_topic_model_inference(user_text)

            if theta_query is None:
                print("Topic Model推理失败，使用简单检索")
                return self._simple_text_retrieval(user_text, top_k)

            # 使用双阶段检索
            candidates = self._dual_stage_retrieval(theta_query, recon_color, top_k)
            return candidates

        except Exception as e:
            print(f"LangChain检索失败: {e}")
            return self._simple_text_retrieval(user_text, top_k)

    def _simple_text_retrieval(self, user_text: str, top_k: int = 10) -> List[Dict]:
        """简单的文本检索作为备用方案"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # 创建简单的TF-IDF向量
            texts = [user_text] + list(self.text_descriptions)
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)

            # 计算相似度
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # 获取Top-K
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            candidates = []
            for idx in top_indices:
                candidates.append({
                    'index': idx,
                    'text_score': similarities[idx],
                    'color_score': 0.5,  # 默认颜色得分
                    'combined_score': similarities[idx] * 0.8 + 0.5 * 0.2,
                    'description': self.text_descriptions[idx],
                    'colors': self.color_vectors[idx].reshape(-1, 3)
                })

            return candidates

        except Exception as e:
            print(f"简单检索也失败: {e}")
            return []

    def run_langchain_pipeline(self, user_text: str, image_path: str = None, top_k: int = 5, num_colors: int = 5) -> Dict[str, Any]:
        """Run LangChain enhanced RAG pipeline with Few-Shot color control"""

        print("🚀 Starting LangChain Topic-RAG System...")

        # Step 0: Detect language and translate if needed
        original_lang = self._detect_language(user_text)
        print(f"Detected language: {original_lang}")

        # Translate user input to English for processing
        if original_lang == "zh":
            user_text_en = self._translate_text(user_text, target_lang="en")
            print("Translated user input to English for processing")
        else:
            user_text_en = user_text

        # Step 1: Retrieve candidate schemes
        print("Step 1: Retrieving candidate schemes...")
        candidates = self._retrieve_candidates_langchain(user_text_en, top_k)

        # Build reference text in English
        reference_text = "Reference Design Schemes (for inspiration only, do not copy):\n\n"
        for i, candidate in enumerate(candidates[:3], 1):
            colors = candidate['colors']
            color_desc = ""
            for j, color in enumerate(colors, 1):
                color_desc += f"Color {j}: RGB({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})\n"

            reference_text += f"Scheme {i}:\nDescription: {candidate['description']}\nColors:\n{color_desc}\n"

        # Step 2: Generate new scheme using LLM
        print("Step 2: Generating new scheme using LLM...")

        try:
            # Check if LLM is available
            if self.llm is None:
                print("LLM not available, generating simple color scheme")
                return self._generate_simple_plan(candidates, user_text_en, original_lang)

            # Build Few-Shot prompt with specified number of colors
            full_prompt = self._create_fewshot_prompt(num_colors, user_text_en, reference_text)

            # Call LLM directly
            generated_plan = self.llm._call(full_prompt)
            print(f"Generated scheme length: {len(generated_plan)}")

            return {
                'generated_plan': generated_plan,
                'candidates': candidates[:3],
                'original_language': original_lang,
                'user_text_original': user_text if original_lang == "zh" else None
            }

        except Exception as e:
            print(f"LangChain processing failed: {e}")
            return self._generate_simple_plan(candidates, user_text_en, original_lang)
    
    def _generate_simple_plan(self, candidates: List[Dict], user_text: str, original_lang: str = "en") -> Dict[str, Any]:
        """Generate simple color scheme (when LLM is not available)"""
        try:
            if not candidates:
                return {
                    'error': 'No suitable reference schemes found',
                    'candidates': []
                }

            # Generate simple scheme based on best candidate
            best_candidate = candidates[0]
            colors = best_candidate['colors']

            # Generate color descriptions
            color_descriptions = []
            color_names = ["Primary Color", "Secondary Color", "Accent Color", "Background Color", "Emphasis Color"]

            for i, color in enumerate(colors[:5]):
                r = int(color[0] * 255)
                g = int(color[1] * 255)
                b = int(color[2] * 255)
                color_name = color_names[i] if i < len(color_names) else f"Color {i+1}"
                color_descriptions.append(f"{color_name}: RGB({r}, {g}, {b})")

            # Generate simple scheme description in English
            simple_plan = f"""**Design Concept**
Based on your requirements "{user_text}", we recommend a carefully matched color scheme. This scheme integrates modern design concepts, maintaining visual harmony while highlighting key elements.

**Color Scheme**
{chr(10).join(color_descriptions)}

**Application Suggestions**
This color scheme is suitable for the application scenarios you mentioned. We recommend using the primary color for large areas, secondary color for functional areas, accent color for highlighting important elements, background color for maintaining spatial cleanliness, and emphasis color for guiding key information.

**Design Features**
- Rich color layers with excellent visual effects
- Complies with modern aesthetic trends
- Balance of practicality and aesthetics
- Easy to apply and match in practice"""

            return {
                'generated_plan': simple_plan,
                'candidates': candidates[:3],
                'note': 'Using simplified generation mode (LLM not available)',
                'original_language': original_lang
            }

        except Exception as e:
            return {
                'error': f'Simple scheme generation failed: {str(e)}',
                'candidates': candidates[:3] if candidates else []
            }

    def get_conversation_history(self) -> List[str]:
        """获取对话历史"""
        return self.memory.chat_memory.messages if self.memory else []

    def clear_memory(self):
        """清除记忆"""
        if self.memory:
            self.memory.clear()

    def _analyze_image_with_doubao(self, image_path: str) -> str:
        """使用豆包API分析图片"""
        try:
            if not self.doubao_api_key:
                return "未配置豆包API密钥，使用默认图片描述"

            from openai import OpenAI
            from PIL import Image
            import base64
            import io

            # 读取并处理图片
            image = Image.open(image_path)

            # 转换为RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 调整图片大小
            max_size = (800, 600)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

            # 编码图片
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=85, optimize=True)
            img_buffer.seek(0)
            image_data = img_buffer.getvalue()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # 设置环境变量
            os.environ["ARK_API_KEY"] = self.doubao_api_key

            # 初始化豆包客户端
            client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=self.doubao_api_key,
            )

            # 调用豆包视觉模型
            response = client.chat.completions.create(
                model="doubao-1-5-thinking-vision-pro-250428",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": "请详细描述这张图片的内容，包括：1.主要物体和场景 2.色彩搭配和风格特点 3.整体氛围和感受 4.可能的设计风格。请用简洁明了的语言描述，重点关注色彩和设计相关的信息。"
                            },
                        ],
                    }
                ],
                timeout=30
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"豆包图片分析失败: {e}")
            return f"图片分析失败: {str(e)}，使用默认描述：现代设计风格，色彩搭配和谐"

    def run_complete_pipeline(self, user_text: str, image_path: str = None, top_k: int = 5, num_colors: int = 5) -> Dict[str, Any]:
        """运行完整的两模块RAG流程（为Streamlit应用提供的接口）"""
        try:
            print("🚀 启动完整的两模块RAG流程...")

            # 模块一：图片理解 + 文本融合 + Topic Model推理
            print("📊 模块一：图片理解与文本融合...")

            # 图片理解（如果提供了图片）
            image_analysis = ""
            if image_path and os.path.exists(image_path):
                try:
                    # 使用豆包API进行图片理解
                    image_analysis = self._analyze_image_with_doubao(image_path)
                    print("✅ 图片分析完成")
                except Exception as e:
                    print(f"图片分析失败: {e}")
                    image_analysis = "图片分析失败，使用默认描述：现代简约设计风格，色彩搭配和谐"

            # 文本融合
            if image_analysis:
                fused_text = f"用户需求: {user_text}\n图片分析: {image_analysis}"
            else:
                fused_text = user_text

            print("📈 Topic Model推理...")

            # Topic Model推理
            recon_text, recon_color, theta_query, mu, logvar = self._complete_topic_model_inference(fused_text)

            if theta_query is None:
                return {
                    'error': 'Topic Model推理失败',
                    'fused_text': fused_text,
                    'image_analysis': image_analysis
                }

            # 模块二：检索 + 重排序 + 生成
            print("🔍 模块二：检索与生成...")

            # 双阶段检索
            candidates = self._dual_stage_retrieval(theta_query, recon_color, top_k)

            if not candidates:
                return {
                    'error': '检索失败，未找到相关候选方案',
                    'fused_text': fused_text,
                    'image_analysis': image_analysis
                }

            # 使用LangChain生成最终方案，传入颜色数量参数
            result = self.run_langchain_pipeline(user_text, image_path, top_k, num_colors)

            # 添加处理过程信息
            result['fused_text'] = fused_text
            result['image_analysis'] = image_analysis
            result['module_one_success'] = True
            result['module_two_success'] = True

            return result

        except Exception as e:
            print(f"完整流程执行失败: {e}")
            return {
                'error': f'完整流程执行失败: {str(e)}',
                'fused_text': user_text,
                'image_analysis': '',
                'module_one_success': False,
                'module_two_success': False
            }

def main():
    """主函数：演示LangChain RAG系统"""
    
    api_key = "sk-3c4ba59c8b094106995821395c7bc60e"
    
    # 初始化系统
    system = LangChainTopicRAGSystem(device='cpu', api_key=api_key)
    
    # 测试用例
    user_text = "我想要一个现代简约风格的配色方案，适合办公环境"
    image_path = "test_image.jpg"
    
    # 运行LangChain增强的RAG流程
    result = system.run_langchain_pipeline(user_text, image_path)
    
    if 'error' not in result:
        print("\n" + "="*60)
        print("✨ LangChain生成的方案 ✨")
        print("="*60)
        print(result['generated_plan'])
        print("="*60)
        
        print(f"\n📊 API调用统计:")
        print(f"总Token数: {result['api_stats']['total_tokens']}")
        print(f"总成本: ${result['api_stats']['total_cost']:.4f}")
        
        print(f"\n📝 图片分析:")
        print(result['image_analysis'])
        
        print(f"\n🔄 融合文本:")
        print(result['fused_text'][:200] + "...")
    else:
        print(f"❌ 处理失败: {result['error']}")

if __name__ == "__main__":
    main() 
