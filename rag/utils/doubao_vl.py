import requests
import os

def upload_image_to_imgbb(image_path, imgbb_api_key, max_retries=3):
    """将本地图片上传到imgbb，返回公网URL"""
    import time

    for attempt in range(max_retries):
        try:
            print(f"📤 上传图片到imgBB (第{attempt + 1}次)...")
            print(f"   图片路径: {image_path}")

            with open(image_path, "rb") as f:
                url = "https://api.imgbb.com/1/upload"
                payload = {
                    "key": imgbb_api_key,
                }
                files = {
                    "image": f,
                }
                response = requests.post(url, data=payload, files=files, timeout=60)

                if response.status_code == 200:
                    image_url = response.json()["data"]["url"]
                    print(f"✅ 图片上传成功: {image_url}")
                    return image_url
                else:
                    error_msg = f"imgbb上传失败: {response.status_code} - {response.text}"
                    print(f"❌ {error_msg}")
                    if attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 5)  # 递增等待时间
                        continue
                    else:
                        raise Exception(error_msg)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ 图片上传异常 (第{attempt + 1}次): {error_msg}")
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 5)
                continue
            else:
                raise Exception(f"图片上传失败，已重试{max_retries}次。最后错误: {error_msg}")

def doubao_vl_image_caption_base64(image_path, api_key, base_url, prompt=None, max_retries=3):
    """使用Base64编码直接调用豆包图片理解API，避免网络问题"""
    import base64
    import time
    from openai import OpenAI

    # 读取图片并转换为Base64
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        print(f"📷 图片转换为Base64成功，大小: {len(base64_image)} 字符")
    except Exception as e:
        return f"图片读取失败: {e}"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120
    )

    if prompt is None:
        prompt = (
            "Please analyze this image as a professional designer and output the following in English:\n"
            "1. Content description (main elements/scenes/objects in the image)\n"
            "2. Design style (e.g., modern, minimal, vintage, natural, be specific)\n"
            "3. Main color scheme and palette suggestions (list main, secondary, accent colors, use RGB or common color names)\n"
            "4. Design elements and layout (e.g., spatial division, materials, decorations)\n"
            "5. Emotional atmosphere (e.g., cozy, calm, energetic, romantic, describe the feeling)\n"
            "Please output in a structured, detailed, and professional way."
        )

    # 获取图片格式
    image_format = image_path.split('.')[-1].lower()
    if image_format == 'jpg':
        image_format = 'jpeg'

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}},
            {"type": "text", "text": prompt}
        ]}
    ]

    # 重试机制
    for attempt in range(max_retries):
        try:
            print(f"🔄 尝试调用豆包API (Base64方式，第{attempt + 1}次)...")

            completion = client.chat.completions.create(
                model="doubao-seed-1-6-250615",
                messages=messages
            )

            result = completion.choices[0].message.content
            print(f"✅ 豆包API调用成功！返回内容长度: {len(result)}")
            return result

        except Exception as e:
            error_msg = str(e)
            print(f"❌ 豆包API调用失败 (第{attempt + 1}次): {error_msg}")

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"⏳ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                break

    return f"豆包API调用失败，已重试{max_retries}次。最后错误: {error_msg}"

def doubao_vl_image_caption(image_url, api_key, base_url, prompt=None, max_retries=3):
    """调用豆包图片理解API，返回图片描述（支持自定义prompt）"""
    import time
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120  # 增加超时时间到2分钟
    )

    if prompt is None:
        prompt = (
            "Please analyze this image as a professional designer and output the following in English:\n"
            "1. Content description (main elements/scenes/objects in the image)\n"
            "2. Design style (e.g., modern, minimal, vintage, natural, be specific)\n"
            "3. Main color scheme and palette suggestions (list main, secondary, accent colors, use RGB or common color names)\n"
            "4. Design elements and layout (e.g., spatial division, materials, decorations)\n"
            "5. Emotional atmosphere (e.g., cozy, calm, energetic, romantic, describe the feeling)\n"
            "Please output in a structured, detailed, and professional way."
        )

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt}
        ]}
    ]

    # 重试机制
    for attempt in range(max_retries):
        try:
            print(f"🔄 尝试调用豆包API (第{attempt + 1}次)...")
            print(f"   图片URL: {image_url}")

            completion = client.chat.completions.create(
                model="doubao-seed-1-6-250615",
                messages=messages
            )

            # 解析返回内容
            result = completion.choices[0].message.content
            print(f"✅ 豆包API调用成功！返回内容长度: {len(result)}")
            return result

        except Exception as e:
            error_msg = str(e)
            print(f"❌ 豆包API调用失败 (第{attempt + 1}次): {error_msg}")

            # 如果是超时错误且还有重试次数，等待后重试
            if attempt < max_retries - 1:
                if "timeout" in error_msg.lower() or "downloading" in error_msg.lower():
                    wait_time = (attempt + 1) * 10  # 递增等待时间
                    print(f"⏳ 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 非超时错误，直接返回
                    break

    # 所有重试都失败，返回错误信息
    return f"豆包API调用失败，已重试{max_retries}次。最后错误: {error_msg}"