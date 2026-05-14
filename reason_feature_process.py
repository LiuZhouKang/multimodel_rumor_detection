import torch
import os
from typing import List
import clip
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_from', type=str, default='weibo', help='数据集来源 (默认: weibo)')
args = parser.parse_args()

# 确保目录存在
os.makedirs("reason_feature", exist_ok=True)

train_text_reason = np.load("reason_content/train_text_reason.npy", allow_pickle=True)
train_image_reason = np.load("reason_content/train_image_reason.npy", allow_pickle=True)
test_text_reason = np.load("reason_content/test_text_reason.npy", allow_pickle=True)
test_image_reason = np.load("reason_content/test_image_reason.npy", allow_pickle=True)

def extract_text_features(texts: List[str], data_from: str, batch_size: int = 64) -> tuple[torch.Tensor, np.ndarray]:
    """
    借助CLIP对输入的中文或英文文本进行处理，提取特征矩阵，并计算每个样本的损失 

    :param texts: 输入的文本列表
    :param data_from: 指定文本语言，'chinese' 或 'english'
    :param batch_size: 批处理大小，默认为64
    :return: 文本的特征矩阵，每个样本的损失数组
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if data_from == 'Twitter':
        # 加载英文CLIP模型
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()

        def tokenize_text(text):
            return clip.tokenize(text, truncate=True).to(device)
            
        def get_text_features(tokens):
            return model.encode_text(tokens)
            
    elif data_from == 'weibo':
        # 加载中文CLIP模型 - 使用 transformers 库
        model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
        
        # 检查本地缓存
        cache_dir = "corpus/chinese-clip-cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        model = ChineseCLIPModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        processor = ChineseCLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        model.eval()

        def get_text_features(texts_batch):
            # 截断文本以避免超过模型的最大长度限制
            inputs = processor(text=texts_batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            text_features = model.get_text_features(**inputs)
            return text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
        def tokenize_text(text):
            # 为了兼容原有的批量处理逻辑，这里返回一个占位符
            # 实际处理在 get_text_features 中完成
            return text
    else:
        raise ValueError("data_from 参数必须为 'weibo' 或 'Twitter'")

    all_features = []
    all_losses = []
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        if data_from == 'weibo':
            # 中文CLIP使用不同的批量处理方式
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
                batch_texts = [str(t) for t in texts[i:i + batch_size]]
                batch_features = get_text_features(batch_texts)
                all_features.append(batch_features.cpu())

                # 计算对比分数
                logits_per_text = batch_features @ batch_features.T / 0.07
                labels = torch.arange(len(batch_texts), device=device)
                loss = criterion(logits_per_text, labels)
                batch_losses = [loss.item()]
                all_losses.extend(batch_losses)
        else:
            # 英文CLIP原有逻辑
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
                batch_texts = texts[i:i + batch_size]
                batch_tokens = torch.cat([tokenize_text(text) for text in batch_texts])
                batch_features = get_text_features(batch_tokens)
                all_features.append(batch_features.cpu())

                logits_per_text = batch_features @ batch_features.T / 0.07
                labels = torch.arange(len(batch_texts), device=device)
                loss = criterion(logits_per_text, labels)
                batch_losses = [loss.item()]
                all_losses.extend(batch_losses)

    return torch.cat(all_features, dim=0), np.array(all_losses)

# 使用示例
if __name__ == "__main__":
    print("\n---------------1.处理训练集中文本理由部分数据---------------")
    train_text_reason_feature, _ = extract_text_features(train_text_reason, data_from=args.data_from)
    
    print("---------------2.处理测试集中文本理由部分数据---------------")
    test_text_reason_feature, _ = extract_text_features(test_text_reason, data_from=args.data_from)
    
    print("---------------3.处理训练集中图像理由部分数据---------------")
    train_image_reason_feature, _ = extract_text_features(train_image_reason, data_from=args.data_from)
    
    print("---------------4.处理测试集中图像理由部分数据---------------")
    test_image_reason_feature, _ = extract_text_features(test_image_reason, data_from=args.data_from)

    # 保存四个列表为 .npy 文件（只保存特征矩阵，不保存损失）
    np.save('reason_feature/train_text_reason_feature.npy', train_text_reason_feature.numpy())
    np.save('reason_feature/test_text_reason_feature.npy', test_text_reason_feature.numpy())
    np.save('reason_feature/train_image_reason_feature.npy', train_image_reason_feature.numpy())
    np.save('reason_feature/test_image_reason_feature.npy', test_image_reason_feature.numpy())

    print("四个列表已保存为 .npy 文件")
