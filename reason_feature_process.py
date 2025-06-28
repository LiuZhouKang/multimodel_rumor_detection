import torch
from typing import List
import clip
from cn_clip.clip import load_from_name, tokenize
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_from', type=str, default='weibo', help='数据集来源 (默认: weibo)')
args = parser.parse_args()

train_text_reason = np.load("reason_content/train_text_reason.npy")
train_image_reason = np.load("reason_content/train_image_reason.npy")
test_text_reason = np.load("reason_content/test_text_reason.npy")
test_image_reason = np.load("reason_content/test_image_reason.npy")

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
    elif data_from == 'weibo':
        # 加载中文CLIP模型
        model, preprocess = load_from_name("ViT-B-16", device=device)
        model.eval()

        def tokenize_text(text):
            return tokenize([text]).to(device)
    else:
        raise ValueError("data_from 参数必须为 'weibo' 或 'Twitter'")

    all_features = []
    all_losses = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
            batch_texts = texts[i:i + batch_size]
            # 对当前批次的文本进行tokenize
            batch_tokens = torch.cat([tokenize_text(text) for text in batch_texts])
            # 提取特征
            batch_features = model.encode_text(batch_tokens)
            all_features.append(batch_features.cpu())

            # 计算对比分数
            logits_per_text = batch_features @ batch_features.T / 0.07  # 0.07 是温度系数，CLIP 常用值
            labels = torch.arange(len(batch_texts), device=device)
            # 计算损失
            loss = criterion(logits_per_text, labels)
            # 为每个样本分配相同的损失值（在实际应用中，也可以根据需求调整）
            batch_losses = [loss.item()]
            all_losses.extend(batch_losses)

    # 合并所有批次的特征
    return torch.cat(all_features, dim=0), np.array(all_losses)

# 使用示例
if __name__ == "__main__":
    # 英文文本示例
    print("\n---------------1.处理训练集中文本理由部分数据---------------")
    train_text_reason_feature = extract_text_features(train_text_reason, data_from=args.data_from)
    # print(train_text_reason_feature.shape)
    print("---------------2.处理测试集中文本理由部分数据---------------")
    test_text_reason_feature = extract_text_features(test_text_reason, data_from=args.data_from)
    # print(test_text_reason_feature.shape)
    print("---------------3.处理训练集中图像理由部分数据---------------")
    train_image_reason_feature = extract_text_features(train_image_reason, data_from=args.data_from)
    # print(train_image_reason_feature.shape)
    print("---------------4.处理测试集中图像理由部分数据---------------")
    test_image_reason_feature = extract_text_features(test_image_reason, data_from=args.data_from)
    # print(test_image_reason_feature.shape)
    

    # 保存四个列表为 .npy 文件
    np.save('reason_feature/train_text_reason_feature.npy', np.array(train_text_reason_feature))
    np.save('reason_feature/test_text_reason_feature.npy', np.array(test_text_reason_feature))
    np.save('reason_feature/train_image_reason_feature.npy', np.array(train_image_reason_feature))
    np.save('reason_feature/test_image_reason_feature.npy', np.array(test_image_reason_feature))

    print("四个列表已保存为 .npy 文件")
    # print(len(train_text_reason))
    # print(len(test_text_reason))
    # print(len(train_image_reason))
    # print(len(test_image_reason))