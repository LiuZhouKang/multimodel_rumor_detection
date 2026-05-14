import requests
import json
import argparse
from data_prepare import *
import base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import os

# 获取对应的待处理数据集
parser = argparse.ArgumentParser()
parser.add_argument('--data_from', type=str, default='weibo', help='数据集来源 (默认: weibo)')
parser.add_argument('--batch_size', type=int, default=32, help='批量小规模数据集大小')
parser.add_argument('--num_workers', type=int, default=8, help='并行线程数')
parser.add_argument('--model_provider', type=str, default='zhipu', choices=['zhipu', 'qwen'], help='大模型提供商 (默认: zhipu)')
parser.add_argument('--api_key', type=str, default='', help='API Key (如果为空则从代码中读取)')
args = parser.parse_args()

if args.data_from == 'weibo':
    train_image_path, train_text, train_label, test_image_path, test_text, test_label = load_weibo_datasets()
    data_from = "weibo"
elif args.data_from == 'Twitter':
    train_image_path, train_text, train_label, test_image_path, test_text, test_label = load_twitter_datasets()
    data_from = "Twitter"

train_text_reason = []
test_text_reason = []
train_image_reason = []
test_image_reason = []

# 定义批量大小
BATCH_SIZE = args.batch_size
MODEL_PROVIDER = args.model_provider

# API Keys配置
ZHIPU_API_KEY = args.api_key if args.api_key else '33b4969db86e4599a8b47a61e2125a0a.WtHV4Q6OjGKXbytd'
QWEN_API_KEY = args.api_key if args.api_key else 'sk-4ba3868d640749f19300ff010ff50a42'

# 初始化对应客户端
if MODEL_PROVIDER == 'zhipu':
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    print(f"使用智谱AI GLM模型进行推理")
elif MODEL_PROVIDER == 'qwen':
    import dashscope
    from dashscope import MultiModalConversation
    dashscope.api_key = QWEN_API_KEY
    print(f"使用通义千问 Qwen 模型进行推理")


def get_messages(content_to_evaluate, is_text, data_from):
    if is_text:
        if data_from == "Twitter":
            messages = [
                {"role": "system", "content": "Given the following news, predict its authenticity and explain the reasons for your prediction. Please avoid providing ambiguous evaluations such as \"undetermined\" or \"uncertain\" or \"unpredictable\"."},
                {"role": "user", "content": content_to_evaluate}
            ]
        elif data_from == "weibo":
            messages = [
                {"role": "system", "content": "给定以下新闻，请预测其真实性，并说明预测的原因。请避免提供如未确定、无法确定、无法预测、不做评价、无法判断等模棱两可的评估。"},
                {"role": "user", "content": content_to_evaluate}
            ]
    else:
        with open(content_to_evaluate, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        if data_from == "Twitter":
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type": "text", "text": "This picture is from a news report. Please describe its content, then determine the authenticity of the picture content and provide the reasons for your judgment. Please avoid giving uncertain or ambiguous evaluations such as \"undetermined\" or \"unable to determine\" or \"unpredictable\"."}
                ]
            }]
        elif data_from == "weibo":
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type": "text", "text": "这张图片来自一篇新闻报道，请根据图片内容判断其真实性，并给出理由。请避免提供如未确定、无法确定、无法预测、不做评价、无法判断等模棱两可的评估。"}
                ]
            }]
    return messages


def call_zhipu_model(messages, is_text):
    """调用智谱AI模型"""
    try:
        if is_text:
            response = client.chat.completions.create(model="glm-4-flash", messages=messages, max_tokens=150)
        else:
            response = client.chat.completions.create(model="glm-4v-flash", messages=messages, max_tokens=150)
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"Failed! Switching to another model... Error: {e}")
        try:
            if is_text:
                response = client.chat.completions.create(model="glm-4", messages=messages, max_tokens=150)
            else:
                response = client.chat.completions.create(model="glm-4v", messages=messages, max_tokens=150)
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(f"Failed again! Error: {e}")
            return None


def call_qwen_model(messages, is_text):
    """调用通义千问模型"""
    try:
        if is_text:
            # 千问文本模型 - 使用Chat API
            from dashscope import Generation
            response = Generation.call(
                model='qwen-turbo',
                messages=messages,
                result_format='message',
            )
        else:
            # 千问多模态模型
            from dashscope import MultiModalConversation
            response = MultiModalConversation.call(
                model='qwen-vl-max',
                messages=messages,
            )
        
        if response.status_code == 200:
            # 处理千问返回格式
            output = response.output
            if hasattr(output, 'choices') and output.choices:
                content = output.choices[0].message.content
                # 处理千问返回格式差异
                if isinstance(content, list):
                    content = content[0].get('text', '') if content else ''
                elif isinstance(content, str):
                    pass  # 已经是字符串
                return content
            return None
        else:
            print(f"Qwen API error: {response.code} - {response.message}")
            return None
    except Exception as e:
        print(f"Qwen call failed! Error: {e}")
        return None


def process_single_content(content_to_evaluate, is_text, is_train, data_from):
    messages = get_messages(content_to_evaluate, is_text, data_from)
    
    # 根据选择的提供商调用对应模型
    if MODEL_PROVIDER == 'zhipu':
        content = call_zhipu_model(messages, is_text)
    elif MODEL_PROVIDER == 'qwen':
        content = call_qwen_model(messages, is_text)
    else:
        content = None
    
    if content is None:
        # 失败时的默认返回
        if data_from == "weibo":
            return "含有敏感内容，无法判断其真实性", is_train, is_text
        elif data_from == "Twitter":
            return "The content contains sensitive information and cannot be evaluated.", is_train, is_text
    
    return content, is_train, is_text


def judge(content_list, is_text, is_train, data_from):
    global train_text_reason, test_text_reason, train_image_reason, test_image_reason
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for content in content_list:
            futures.append(executor.submit(process_single_content, content, is_text, is_train, data_from))

        for future in tqdm(futures, desc=f"Processing data (Provider: {MODEL_PROVIDER})"):
            result, is_train, is_text = future.result()
            if result is not None:
                if is_train:
                    if is_text:
                        train_text_reason.append(result)
                    else:
                        train_image_reason.append(result)
                else:
                    if is_text:
                        test_text_reason.append(result)
                    else:
                        test_image_reason.append(result)
            else:
                print("Failed to get result after retries. Skipping...")


# 使用示例
if __name__ == "__main__":
    print(f"\n=============== 使用 {MODEL_PROVIDER.upper()} 提供商进行推理 ===============")
    print("\n---------------1.处理训练集中文本部分数据---------------")
    judge(train_text, is_text=True, is_train=True, data_from=data_from)
    print("---------------2.处理测试集中文本部分数据---------------")
    judge(test_text, is_text=True, is_train=False, data_from=data_from)
    print("---------------3.处理训练集中图像部分数据---------------")
    judge(train_image_path, is_text=False, is_train=True, data_from=data_from)
    print("---------------4.处理测试集中图像部分数据---------------")
    judge(test_image_path, is_text=False, is_train=False, data_from=data_from)

    # 确保目录存在
    os.makedirs('reason_content', exist_ok=True)

    # 确保目录存在
    os.makedirs("~/multimodel_rumor_detection/reason_content", exist_ok=True)
    
    # 保存推理结果
    np.save('~/multimodel_rumor_detection/reason_content/train_text_reason.npy', np.array(train_text_reason))
    np.save('~/multimodel_rumor_detection/reason_content/test_text_reason.npy', np.array(test_text_reason))
    np.save('~/multimodel_rumor_detection/reason_content/train_image_reason.npy', np.array(train_image_reason))
    np.save('~/multimodel_rumor_detection/reason_content/test_image_reason.npy', np.array(test_image_reason))

    print("四个列表已保存为 .npy 文件")
