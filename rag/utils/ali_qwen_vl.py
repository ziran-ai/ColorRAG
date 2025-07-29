import requests
import os

def upload_image_to_imgbb(image_path, imgbb_api_key):
    """将本地图片上传到imgbb，返回公网URL"""
    with open(image_path, "rb") as f:
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": imgbb_api_key,
        }
        files = {
            "image": f,
        }
        response = requests.post(url, data=payload, files=files)
        if response.status_code == 200:
            return response.json()["data"]["url"]
        else:
            raise Exception(f"imgbb上传失败: {response.text}")

def ali_qwen_vl_image_caption(image_url, api_key, base_url, prompt=None):
    """调用阿里通义千问图片理解API，返回图片描述（支持自定义prompt）"""
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    if prompt is None:
        prompt = (
            "请用中文、以专业设计师的视角，详细分析这张图片，输出以下内容：\n"
            "1. 内容描述（图片中包含哪些主要元素/场景/物品）\n"
            "2. 设计风格（如现代、极简、复古、自然等，尽量具体）\n"
            "3. 主色调及配色建议（列出主色、辅助色、点缀色，并用RGB或常见色名描述）\n"
            "4. 设计元素与布局（如空间分区、材质、装饰等）\n"
            "5. 情感氛围（如温馨、冷静、活力、浪漫等，描述营造的感受）\n"
            "请分条输出，语言专业、细致、具体。"
        )
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt}
        ]}
    ]
    completion = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=messages
    )
    # 解析返回内容
    try:
        return completion.choices[0].message.content
    except Exception:
        return str(completion) 