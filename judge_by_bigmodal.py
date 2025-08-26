import requests
import json
import argparse
from data_prepare import *
import base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from zhipuai import ZhipuAI
import os

if not os.path.exists('reason_content'):
    os.makedirs('reason_content')
    
# 获取对应的待处理数据集
parser = argparse.ArgumentParser()
parser.add_argument('--data_from', type=str, default='weibo', help='数据集来源 (默认: weibo)')
parser.add_argument('--batch_size', type=int, default=32, help='批量小规模数据集大小')
parser.add_argument('--num_workers', type=int, default=8, help='并行线程数')
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

client = ZhipuAI(api_key='33b4969db86e4599a8b47a61e2125a0a.WtHV4Q6OjGKXbytd') # 填写您自己的APIKey

def get_messages(content_to_evaluate, is_text, data_from):
    if is_text:
        # 请求处理英文文本负载
        if data_from == "Twitter":
            messages = [
                {"role": "system", "content": "Given the following news, predict its authenticity and explain the reasons for your prediction. Please avoid providing ambiguous evaluations such as \"undetermined\" or \"uncertain\" or \"unpredictable\"."},
                {"role": "user", "content": content_to_evaluate}  # 用户输入的待验证文本
            ]
        # 请求处理中文文本负载
        elif data_from == "weibo":
            messages = [
                {"role": "system", "content": "给定以下新闻，请预测其真实性, 并说明预测的原因。请避免提供如“未确定”“无法确定”“无法预测”“不做评价”等模棱两可的评估。"},
                {"role": "user", "content": content_to_evaluate}  # 用户输入的待验证文本
            ]
    else:
        # 将图片进行base64编码
        with open(content_to_evaluate, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        # 请求处理英文图像负载
        if data_from == "Twitter":
            messages = [{
                "role": "user",
                "content": [
                    # 修改为正确的格式
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type": "text", "text": "This picture is from a news report. Please describe its content, then determine the authenticity of the picture content and provide the reasons for your judgment. Please avoid giving uncertain or ambiguous evaluations such as \"undetermined\" or \"unable to determine\" or \"unpredictable\"."}
                ]
            }]
        # 请求处理中文图像负载
        elif data_from == "weibo":
            messages = [{
                "role": "user",
                "content": [
                    # 修改为正确的格式
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type": "text", "text": "这张图片来自一篇新闻报道，请根据图片内容判断其真实性，并给出理由。请避免提供如“未确定”“无法确定”“无法预测”“不做评价”等模棱两可的评估。"}
                ]
            }]
    return messages

def process_single_content(content_to_evaluate, is_text, is_train, data_from):
    messages = get_messages(content_to_evaluate, is_text, data_from)
    try:
        # 发送请求到API
        if is_text:
            response = client.chat.completions.create(model="glm-4-flash", messages=messages, max_tokens=150)
        else:
            response = client.chat.completions.create(model="glm-4v-flash", messages=messages, max_tokens=150)
        content = response.choices[0].message.content
        return content, is_train, is_text
    except Exception as e:
        print("Failed! Switching to another model to try again...")
        try:
            # 遇到敏感内容时，切换模型去处理
            if is_text:
                response = client.chat.completions.create(model="glm-4", messages=messages, max_tokens=150)
            else:
                response = client.chat.completions.create(model="glm-4v", messages=messages, max_tokens=150)
            content = response.choices[0].message.content
            return content, is_train, is_text
        except Exception as e:
            print("Failed again! Skipping this content...")
            if data_from == "weibo":
                return "含有敏感内容，无法判断其真实性", is_train, is_text
            elif data_from == "Twitter":
                return "The content contains sensitive information and cannot be evaluated.", is_train, is_text

def judge(content_list, is_text, is_train, data_from):
    global train_text_reason, test_text_reason, train_image_reason, test_image_reason
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for content in content_list:
            futures.append(executor.submit(process_single_content, content, is_text, is_train, data_from))

        for future in tqdm(futures, desc="Processing data"):
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
    print("\n---------------1.处理训练集中文本部分数据---------------")
    judge(train_text, is_text=True, is_train=True, data_from=data_from)
    print("---------------2.处理测试集中文本部分数据---------------")
    judge(test_text, is_text=True, is_train=False, data_from=data_from)
    print("---------------3.处理训练集中图像部分数据---------------")
    judge(train_image_path, is_text=False, is_train=True, data_from=data_from)
    print("---------------4.处理测试集中图像部分数据---------------")
    judge(test_image_path, is_text=False, is_train=False, data_from=data_from)

    # 保存四个列表为 .npy 文件
    np.save('reason_content/train_text_reason.npy', np.array(train_text_reason))
    np.save('reason_content/test_text_reason.npy', np.array(test_text_reason))
    np.save('reason_content/train_image_reason.npy', np.array(train_image_reason))
    np.save('reason_content/test_image_reason.npy', np.array(test_image_reason))

    print("四个列表已保存为 .npy 文件")
    # print(len(train_text_reason))
    # print(len(test_text_reason))
    # print(len(train_image_reason))
    # print(len(test_image_reason))