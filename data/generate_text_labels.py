import pandas as pd
import os
from dotenv import load_dotenv
import time
import requests
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading

# 加载环境变量
load_dotenv()

# 从环境变量中获取 API 密钥
API_KEY = os.getenv("DEEPSEEK_API_KEY")
# 创建一个锁，用于线程安全的打印和DataFrame操作
print_lock = threading.Lock()
df_lock = threading.Lock()

def generate_text(prompt):
    """
    调用 LLM API 生成文本
    """
    api_url = 'https://api.deepseek.com/v1/chat/completions'  # DeepSeek API 地址
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            response_json = response.json()
            # 根据 DeepSeek API 的实际返回格式提取文本
            return response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        else:
            with print_lock:
                print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        with print_lock:
            print(f"API调用异常: {e}")
        return None

def get_description_from_llm(rgb_colors, palette_name=None):
    """
    使用LLM为5个RGB颜色组成的调色板生成丰富的描述（英文版）。

    Args:
        rgb_colors (list of tuples): 包含5个RGB元组的列表, e.g., [(r1,g1,b1), ...]
        palette_name (str, optional): 调色板的名称，由艺术家提供
    """
    # 为每个RGB颜色添加一个简单的颜色名称描述
    color_descriptions = []
    for i, (r, g, b) in enumerate(rgb_colors):
        # 将RGB值从[0,1]范围转换到[0,255]范围
        r_255 = int(r * 255) if 0 <= r <= 1 else int(r)
        g_255 = int(g * 255) if 0 <= g <= 1 else int(g)
        b_255 = int(b * 255) if 0 <= b <= 1 else int(b)
        
        # 使用转换后的RGB值
        color_descriptions.append(f"Color {i+1} - RGB({r_255}, {g_255}, {b_255})")
    
    # 构建提示词，加入调色板名称（如果有）
    name_part = f"Palette name: \"{palette_name}\"\n\n" if palette_name else ""
    
    prompt = f"""
You are a professional art critic and color theorist. Please analyze the following palette consisting of 5 colors:

{name_part}{color_descriptions[0]}
{color_descriptions[1]}
{color_descriptions[2]}
{color_descriptions[3]}
{color_descriptions[4]}

Based on these colors and the palette name (if provided), please provide a concise yet expressive description covering the following aspects:
1. **Mood & Emotion:** What feelings does this palette evoke? (e.g., serenity, energy, melancholy, nostalgia, futuristic).
2. **Theme & Context:** What themes or settings does this palette suggest? (e.g., "autumn forest", "deep ocean", "urban nightlife", "ancient desert", "digital glitch").
3. **Harmony & Contrast:** Briefly describe the relationship between the colors. Are they harmonious, highly contrasting, soft, or vibrant?

Strictly adhere to the following rules (this is mandatory):
- Your description must explicitly mention all 5 colors and their exact RGB values.
- When mentioning "Color 1", you must immediately follow it with its RGB value, e.g., "Color 1(RGB(120, 180, 210))".
- When mentioning "Color 2", you must immediately follow it with its RGB value, e.g., "Color 2(RGB(20, 40, 80))".
- And so on for all 5 colors.
- Do not use abstract color names in place of "Color 1", "Color 2", etc.; maintain this numbering style with RGB values.
- If a palette name is provided, consider the theme or emotion implied by this name in your description.

Example output:
"This palette consists of Color 1(RGB(120, 180, 210)), Color 2(RGB(20, 40, 80)), Color 3(RGB(240, 100, 90)), Color 4(RGB(210, 180, 140)), and Color 5(RGB(160, 220, 190)). Together, these colors create a serene beach resort atmosphere, evoking images of sunny beaches meeting azure waters. Color 1(RGB(120, 180, 210)) and Color 2(RGB(20, 40, 80)) form a depth contrast, symbolizing different depths of seawater; Color 3(RGB(240, 100, 90)) provides a vivid accent, like coral reefs under the sea; Color 4(RGB(210, 180, 140)) represents the sandy beach, which together with Color 5(RGB(160, 220, 190)) of coastal vegetation, forms a harmonious and lively coastal landscape."

Please integrate the above points into a descriptive paragraph, ensuring that all 5 colors and their exact RGB values are included in your description.
"""
    
    description = generate_text(prompt)
    return description  # 直接返回生成的描述

def process_single_row(args):
    """处理单行数据的函数，用于多线程处理"""
    index, row, has_names = args
    
    try:
        # 提取RGB颜色
        colors = []
        for i in range(1, 6):
            r = row[f'Color_{i}_R']
            g = row[f'Color_{i}_G']
            b = row[f'Color_{i}_B']
            colors.append((r, g, b))
        
        with print_lock:
            print(f"处理索引 {row['row_index']} - 提取的颜色: {colors}")
            print(f"注意：如果RGB值在0到1之间，在发送给LLM前会转换为0到255之间的整数")
        
        # 提取调色板名称（如果有）
        palette_name = None
        if has_names:
            palette_name = row['names']
            # 处理可能的列表格式（如果names存储为字符串形式的列表）
            if isinstance(palette_name, str) and palette_name.startswith('[') and palette_name.endswith(']'):
                try:
                    name_list = eval(palette_name)  # 将字符串形式的列表转换为实际列表
                    if isinstance(name_list, list) and name_list:
                        palette_name = ', '.join(name_list)  # 将列表中的名称用逗号连接
                except:
                    pass  # 如果转换失败，保持原样
            
            with print_lock:
                print(f"调色板名称: {palette_name}")
        
        # 生成描述
        description = get_description_from_llm(colors, palette_name)
        
        if description:
            # 将新处理的行添加到结果中
            row_dict = row.to_dict()
            row_dict['description'] = description
            
            with print_lock:
                print(f"索引 {row['row_index']} 生成的描述: {description[:100]}...")
            return row_dict
        else:
            with print_lock:
                print(f"警告：索引 {row['row_index']} 生成描述失败")
            return None
    
    except Exception as e:
        with print_lock:
            print(f"处理索引 {row['row_index']} 时出错: {e}")
        return None

def process_data(input_file, output_file, test_mode=False, max_workers=3):
    """
    处理数据并生成描述
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        test_mode: 如果为True，则只处理前5个样本
        max_workers: 最大工作线程数
    """
    # 检查文件扩展名并读取数据
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        raise ValueError("输入文件必须是 .xlsx 或 .csv 格式")
    
    # 添加行索引列（从1开始）
    df['row_index'] = range(1, len(df) + 1)
    
    print(f"读取了 {len(df)} 条数据")
    print(f"列名: {list(df.columns)}")
    
    # 如果是测试模式，只取前5个样本
    if test_mode:
        df = df.head(5)
        print("测试模式：只处理前5个样本")
    
    # 如果输出文件已存在，检查哪些索引已经处理过
    processed_indices = set()
    processed_df = pd.DataFrame()
    if os.path.exists(output_file):
        try:
            if output_file.endswith('.csv'):
                processed_df = pd.read_csv(output_file)
            else:
                processed_df = pd.read_excel(output_file)
            
            # 如果输出文件中有row_index和description列，收集已处理的索引
            if 'row_index' in processed_df.columns and 'description' in processed_df.columns:
                processed_indices = set(processed_df['row_index'].dropna().astype(int).tolist())
                print(f"发现已处理的数据，共 {len(processed_indices)} 条")
                print(f"已处理的索引范围: {min(processed_indices) if processed_indices else 0} - {max(processed_indices) if processed_indices else 0}")
            else:
                processed_df = pd.DataFrame()
                print("输出文件不包含必要的列，将重新创建")
        except Exception as e:
            print(f"读取输出文件时出错: {e}")
            processed_df = pd.DataFrame()
    
    # 检查数据中是否有标准的颜色列
    expected_columns = []
    for i in range(1, 6):
        for c in ['R', 'G', 'B']:
            expected_columns.append(f'Color_{i}_{c}')
    
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"警告：缺少以下颜色列: {missing_columns}")
        return
    else:
        print("找到所有标准颜色列")
    
    # 检查是否有names列
    has_names = 'names' in df.columns
    if has_names:
        print("找到'names'列，将用于生成描述")
    else:
        print("警告：未找到'names'列，将只使用颜色信息生成描述")
    
    # 创建一个集合来跟踪正在处理的索引，以及一个字典记录哪个线程处理哪个索引（用于调试）
    currently_processing = set()
    thread_processing = {}
    # 创建一个锁来保护currently_processing集合
    processing_lock = threading.Lock()
    
    # 过滤出尚未处理的行
    df_to_process = df[~df['row_index'].isin(processed_indices)]
    print(f"需要处理 {len(df_to_process)} 条新数据")
    
    if len(df_to_process) == 0:
        print("所有数据已处理完毕，无需进一步处理")
        return
    
    # 计时和进度跟踪
    total_rows = len(df_to_process)
    processed_rows = 0
    avg_time_per_sample = 1.0  # 初始估计每个样本的处理时间（秒）
    start_time = time.time()
    last_save_time = start_time
    
    # 保存结果
    new_results = []  # 存储新处理的结果
    results_lock = threading.Lock()
    
    # 如果处于测试模式，不使用线程池
    if test_mode or total_rows <= 5:
        for _, row in df_to_process.iterrows():
            row_index = row['row_index']
            
            # 检查是否已处理或正在处理
            with processing_lock:
                if row_index in processed_indices:
                    print(f"跳过已处理的行 {row_index}")
                    continue
                if row_index in currently_processing:
                    print(f"警告：行 {row_index} 已经在处理中，跳过")
                    continue
                currently_processing.add(row_index)
                thread_processing[row_index] = "主线程"
                print(f"主线程开始处理行 {row_index}")
            
            try:
                result = process_single_row((_, row, has_names))
                if result:
                    with results_lock:
                        new_results.append(result)
                        processed_rows += 1
                
                # 计算平均处理时间
                current_time = time.time()
                elapsed = current_time - start_time
                if processed_rows > 0:
                    avg_time_per_sample = elapsed / processed_rows
                    
                # 保存中间结果
                if current_time - last_save_time > 60 or processed_rows % 10 == 0:
                    with results_lock:
                        if new_results:  # 只有在有新结果时才保存
                            save_results(new_results, processed_df, output_file, processed_rows, total_rows, avg_time_per_sample)
                            # 清空已保存的结果
                            new_results = []
                    last_save_time = current_time
            
            finally:
                # 无论处理成功与否，都从正在处理集合中移除
                with processing_lock:
                    currently_processing.discard(row_index)
                    if row_index in thread_processing:
                        del thread_processing[row_index]
                    print(f"主线程完成处理行 {row_index}")
    else:
        # 修改为使用队列来控制任务分发
        from queue import Queue
        task_queue = Queue()
        
        # 将所有任务添加到队列
        for _, row in df_to_process.iterrows():
            task_queue.put((_, row))
        
        # 定义工作线程函数
        def worker(worker_id):
            nonlocal processed_rows, avg_time_per_sample, last_save_time
            
            while not task_queue.empty():
                try:
                    # 非阻塞方式获取任务
                    try:
                        index, row = task_queue.get(block=False)
                    except:
                        # 队列可能已空
                        break
                    
                    row_index = row['row_index']
                    
                    # 检查是否已处理或正在处理（带锁的双重检查）
                    with processing_lock:
                        if row_index in processed_indices:
                            print(f"线程{worker_id}跳过已处理的行 {row_index}")
                            task_queue.task_done()
                            continue
                        if row_index in currently_processing:
                            print(f"警告：线程{worker_id}发现行 {row_index} 已经在处理中(被{thread_processing.get(row_index, '未知线程')})，跳过")
                            task_queue.task_done()
                            continue
                        
                        # 加入正在处理集合
                        currently_processing.add(row_index)
                        thread_processing[row_index] = f"线程{worker_id}"
                        print(f"线程{worker_id}开始处理行 {row_index}")
                    
                    try:
                        result = process_single_row((index, row, has_names))
                        if result:
                            with results_lock:
                                new_results.append(result)
                                processed_rows += 1
                                
                                # 保存进度
                                current_time = time.time()
                                elapsed = current_time - start_time
                                if processed_rows > 0:
                                    avg_time_per_sample = elapsed / processed_rows
                                
                                # 每处理10行或每60秒保存一次
                                if current_time - last_save_time > 60 or processed_rows % 10 == 0:
                                    if new_results:  # 只有在有新结果时才保存
                                        temp_results = new_results.copy()
                                        new_results.clear()  # 清空已保存的结果，使用clear()而不是重新赋值
                                        save_results(temp_results, processed_df, output_file, processed_rows, total_rows, avg_time_per_sample)
                                    last_save_time = current_time
                    
                    finally:
                        # 无论处理成功与否，都从正在处理集合中移除
                        with processing_lock:
                            currently_processing.discard(row_index)
                            if row_index in thread_processing:
                                del thread_processing[row_index]
                            print(f"线程{worker_id}完成处理行 {row_index}")
                        task_queue.task_done()
                
                except Exception as e:
                    print(f"线程{worker_id}出错: {e}")
        
        # 创建并启动工作线程
        threads = []
        for i in range(max_workers):
            t = threading.Thread(target=worker, args=(i+1,))
            t.daemon = True
            t.start()
            threads.append(t)
        
        # 等待所有任务完成
        for t in threads:
            t.join()
    
    # 最终保存剩余结果
    with results_lock:
        if new_results:
            save_results(new_results, processed_df, output_file, processed_rows, total_rows, avg_time_per_sample)
    
    # 最后检查是否有索引仍在处理中（不应该发生）
    with processing_lock:
        if currently_processing:
            print(f"警告：程序结束时仍有 {len(currently_processing)} 个索引在处理中: {sorted(currently_processing)[:10]}")
    
    total_time = time.time() - start_time
    print(f"所有描述生成完毕！结果已保存到 {output_file}")
    print(f"总处理时间: {total_time/60:.1f}分钟，平均每个样本: {avg_time_per_sample:.2f}秒")

def save_results(new_results, processed_df, output_file, processed_rows, total_rows, avg_time_per_sample):
    """保存结果到文件"""
    if not new_results:
        return
    
    print(f"正在保存 {len(new_results)} 个新结果...")
    
    # 合并结果到DataFrame
    with df_lock:
        # 创建新结果的DataFrame
        new_df = pd.DataFrame(new_results)
        
        # 合并到已处理的DataFrame
        if not processed_df.empty:
            # 防止重复，先删除可能重复的行
            if 'row_index' in processed_df.columns and 'row_index' in new_df.columns:
                # 获取新的索引
                new_indices = set(new_df['row_index'].astype(int).tolist())
                
                # 从已有DataFrame中删除重复的行（如果有）
                processed_df = processed_df[~processed_df['row_index'].isin(new_indices)]
            
            # 拼接数据
            all_df = pd.concat([processed_df, new_df], ignore_index=True)
        else:
            all_df = new_df
        
        # 确保按索引排序
        if 'row_index' in all_df.columns:
            all_df = all_df.sort_values(by='row_index')
    
    # 保存到文件
    try:
        # 不再对processed_df做in-place修改，而是返回新的DataFrame
        if output_file.endswith('.csv'):
            all_df.to_csv(output_file, index=False)
        else:
            all_df.to_excel(output_file, index=False)
        
        print(f"进度: {processed_rows}/{total_rows} ({processed_rows/total_rows*100:.2f}%) 已保存。")
        print(f"当前已处理总行数: {len(all_df)}")
        print(f"预计剩余时间: {(total_rows-processed_rows)*avg_time_per_sample/60:.1f}分钟")
        
        # 将合并后的DataFrame返回，确保后续操作使用最新的数据
        return all_df
    except Exception as e:
        print(f"保存时出错: {e}")
        # 尝试保存为不同格式
        try:
            base, ext = os.path.splitext(output_file)
            backup_file = f"{base}_backup{ext}"
            if ext.lower() == '.xlsx':
                all_df.to_csv(f"{base}.csv", index=False)
                print(f"已保存备份到CSV格式: {base}.csv")
            else:
                all_df.to_excel(backup_file, index=False)
                print(f"已保存备份: {backup_file}")
        except Exception as e2:
            print(f"保存备份也失败: {e2}")
        return processed_df  # 发生错误时返回原始DataFrame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成色彩方案的文本描述')
    parser.add_argument('--input', type=str, default='data/dataset_color.xlsx', help='输入文件路径')
    parser.add_argument('--output', type=str, default='data/palettes_with_descriptions.xlsx', help='输出文件路径')
    parser.add_argument('--test', action='store_true', help='测试模式，只处理前5个样本')
    parser.add_argument('--threads', type=int, default=3, help='并行处理的线程数（默认为3）')
    
    args = parser.parse_args()
    
    # 检查API密钥是否设置
    if not API_KEY:
        print("错误：未找到API密钥，请在.env文件中设置DEEPSEEK_API_KEY")
        exit(1)
    
    process_data(args.input, args.output, args.test, args.threads) 