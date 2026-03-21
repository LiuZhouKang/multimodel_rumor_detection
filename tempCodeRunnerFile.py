import clip
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_prepare import *

# 获取对应的待处理数据集
parser = argparse.ArgumentParser()
parser.add_argument('--data_from', type=str, default='weibo', help='数据集来源 (默认: weibo)')
args = parser.parse_args()
    
if args.data_from=='weibo':
    train_image_path,train_text,train_label,test_image_path,test_text,test_label=load_weibo_datasets()
elif args.data_from=='Twitter':
    train_image_path,train_text,train_label,test_image_path,test_text,test_label=load_twitter_datasets()

class TextDataset(Dataset):
    """自定义文本数据集类"""
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx])

class ImageDataset(Dataset):
    """自定义图像数据集类"""
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        return self.preprocess(image)

def clip_texts(texts, batch_size=64, model_name="ViT-B/32"):
    """
    批量处理文本获取视觉增强的特征向量
    
    参数:
        texts: list[str] - 文本列表
        batch_size: int - 批处理大小
        model_name: str - CLIP模型名称
    
    返回:
        torch.Tensor - 所有文本的特征矩阵 (num_texts, feature_dim)
    """
    
    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(model_name, device=device)
    model.eval()
    
    # 创建数据加载器
    def tokenize_with_truncate(text):
        # 确保返回2D张量 (1, sequence_length)
        tokens = clip.tokenize(text, truncate=True).squeeze(0)  # 移除多余的batch维度
        return tokens
    
    dataset = TextDataset(texts, tokenize_with_truncate)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: torch.stack(x).to(device)  # 这里会自动添加batch维度
    )
    
    # 批量处理
    all_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing texts"):
            batch_features = model.encode_text(batch)
            all_features.append(batch_features.cpu())
    
    # 合并所有批次
    return torch.cat(all_features, dim=0)

def clip_images(image_paths, batch_size=64, model_name="ViT-B/32"):
    """
    批量处理图像获取文本增强的特征向量
    
    参数:
        image_paths: list[str] - 图像路径列表
        batch_size: int - 批处理大小
        model_name: str - CLIP模型名称
    
    返回:
        torch.Tensor - 所有图像的特征矩阵 (num_images, feature_dim)
    """
    
    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    
    # 创建数据加载器
    dataset = ImageDataset(image_paths, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4  # 使用多线程加速图像加载
    )
    
    # 批量处理
    all_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            batch = batch.to(device)
            batch_features = model.encode_image(batch)
            all_features.append(batch_features.cpu())
    
    return torch.cat(all_features, dim=0)

if __name__ == "__main__":
    print("\n------------1.获得训练集视觉增强后的文本特征中------------")
    train_text_feature=clip_texts(train_text)
    #print(len(train_text_feature))
    print("------------2.获得测试集视觉增强后的文本特征中------------")
    test_text_feature=clip_texts(test_text)
    # print(len(test_text_feature))
    print("------------3.获得训练集文本增强后的图像特征中------------")
    train_image_feature=clip_images(train_image_path)
    #print(len(train_image_feature))
    print("------------4.获得测试集文本增强后的图像特征中------------")
    test_image_feature=clip_images(test_image_path)
    # print(len(test_image_feature))
    
    #print(f"训练集视觉增强后文本特征向量矩阵形状: {train_text_feature.shape}")
    #print(f"测试集视觉增强后文本特征向量矩阵形状: {test_text_feature.shape}")
    #print(f"训练集视觉增强后图像特征向量矩阵形状: {train_image_feature.shape}")
    #print(f"测试集视觉增强后图像特征向量矩阵形状: {test_image_feature.shape}")
    
    print("--------------5.处理完毕，正在保存处理后特征--------------")
    # 保存特征
    np.save("clip_feature/train_text_clip_feature.npy", train_text_feature.numpy())
    np.save("clip_feature/test_text_clip_feature.npy", test_text_feature.numpy())
    np.save("clip_feature/train_image_clip_feature.npy", train_image_feature.numpy())
    np.save("clip_feature/test_image_clip_feature.npy", test_image_feature.numpy())
    np.save("clip_feature/train_label.npy", train_label)
    np.save("clip_feature/test_label.npy", test_label)