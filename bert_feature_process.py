import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from data_prepare import * 

# 获取对应的待处理数据集
parser = argparse.ArgumentParser()
parser.add_argument('--data_from', type=str, default='weibo', help='数据集来源 (默认: weibo)')
args = parser.parse_args()
    
if args.data_from=='weibo':
    train_image_path,train_text,train_label,test_image_path,test_text,test_label=load_weibo_datasets()
    model_name='corpus/chinese_L-12_H-768_A-12'
elif args.data_from=='Twitter':
    train_image_path,train_text,train_label,test_image_path,test_text,test_label=load_twitter_datasets()
    model_name='corpus/uncased_L-12_H-768_A-12'

# 对Bert提取到的文本特征矩阵进行形状转换，得到与CLIP提取到的特征矩阵相同规模的特征矩阵
class Text_Bert_Feature_Extractor(nn.Module):
    def __init__(self):
        super(Text_Bert_Feature_Extractor, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.relu(x)
        return x

def extract_bert_features(model_name='corpus/chinese_L-12_H-768_A-12', max_length=512):
    """
    使用BERT模型提取文本特征矩阵
    
    参数:
        model_name: str - BERT模型名称
        max_length: int - 最大文本长度
    
    返回:
        np.ndarray - 文本特征矩阵 (num_texts, feature_dim)
    """
    # 初始化模型和tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device).eval()
    
    # 批量处理文本
    print("\n------------1.获得训练集Bert处理后的文本特征中------------")
    train_text_feature = []
    with torch.no_grad():
        for text in tqdm(train_text, desc="Processing texts"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding='max_length'
            ).to(device)
            
            outputs = model(**inputs)
            # 使用[CLS] token的特征作为文本表示
            text_feature = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            train_text_feature.append(text_feature)
    
    print("------------2.获得测试集Bert处理后的文本特征中------------")
    test_text_feature = []
    with torch.no_grad():
        for text in tqdm(test_text, desc="Processing texts"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding='max_length'
            ).to(device)

            outputs = model(**inputs)
            # 使用[CLS] token的特征作为文本表示
            text_feature = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            test_text_feature.append(text_feature)
    
    # 对文本特征矩阵进行维度转换
    # 创建模型
    feature_extractor = Text_Bert_Feature_Extractor()
    # 前向传播，得到最后结果
    train_text_feature = feature_extractor(torch.tensor(train_text_feature))
    test_text_feature = feature_extractor(torch.tensor(test_text_feature))

    # 保存为numpy矩阵
    print("--------------3.处理完毕，正在保存处理后特征--------------")
    train_text_feature = train_text_feature.detach().numpy()  # 使用detach()取消梯度
    np.save("normal_feature/train_text_Bert_feature.npy", train_text_feature)
    test_text_feature = test_text_feature.detach().numpy()  # 使用detach()取消梯度
    np.save("normal_feature/test_text_Bert_feature.npy", test_text_feature)

if __name__ == "__main__":
    print("--------------一.利用Bert对文本特征进行提取---------------")
    extract_bert_features(model_name)
