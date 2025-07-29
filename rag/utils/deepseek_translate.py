import os
import requests

def deepseek_translate(text, target_lang="en", api_key=None):
    """
    使用DeepSeek API进行中英互译。
    target_lang: "en"（翻译成英文）或 "zh"（翻译成中文）
    """
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "sk-3c4ba59c8b094106995821395c7bc60e")
    url = "https://api.deepseek.com/v1/chat/completions"
    if target_lang == "en":
        prompt = f"Translate the following text to English, keep the meaning and style:\n{text}"
    else:
        prompt = f"请将以下内容翻译成中文，保持原意和风格：\n{text}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.2
    }
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip() 