import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
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

# 对VGG提取到的图像特征矩阵进行形状转换，得到与CLIP提取到的特征矩阵相同规模的特征矩阵
class Image_VGG_Feature_Extractor(nn.Module):
    def __init__(self):
        super(Image_VGG_Feature_Extractor, self).__init__()
        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.relu(x)
        return x

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
    model = Text_Bert_Feature_Extractor()
    # 前向传播，得到最后结果
    train_text_feature = model(torch.tensor(train_text_feature))
    test_text_feature = model(torch.tensor(test_text_feature))

    # 保存为numpy矩阵
    print("--------------3.处理完毕，正在保存处理后特征--------------")
    train_text_feature = train_text_feature.detach().numpy()  # 使用detach()取消梯度
    # print(train_text_feature.shape)
    np.save("normal_feature/train_text_Bert_feature.npy", train_text_feature)
    test_text_feature = test_text_feature.detach().numpy()  # 使用detach()取消梯度
    # print(test_text_feature.shape)
    np.save("normal_feature/test_text_Bert_feature.npy", test_text_feature)

def extract_vgg_features(layer_index=5):
    """
    使用VGG19提取图像特征矩阵
    
    参数:
        layer_index: int - 要提取的特征层索引(默认第5层)
    """
    # 初始化VGG模型
    vgg = models.vgg19(pretrained=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = vgg.to(device)
    
    # 截取指定层之前的网络
    feature_extractor = torch.nn.Sequential(*list(vgg.features.children())[:layer_index+1])
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # 提取特征
    # 批量处理函数
    def batch_process(image_paths, batch_size=16):
        features = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i+batch_size]
            
            # 批量加载图像
            batch_images = []
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                batch_images.append(transform(img))
                
            # 批量处理
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                batch_features = feature_extractor(batch_tensor)
                features.extend([f.cpu().numpy() for f in batch_features])
                      
        return np.array(features)

    # 创建矩阵规模转换模型
    model = Image_VGG_Feature_Extractor()

    # 使用批量处理
    print("------------1.获得训练集VGG处理后的图像特征中-------------")
    train_image_feature = batch_process(train_image_path)
    print("------------2.获得测试集VGG处理后的图像特征中-------------") 
    test_image_feature = batch_process(test_image_path)
    # print(test_image_feature.shape)
    
    # 保存为numpy矩阵
    print("--------------3.处理完毕，正在保存处理后特征--------------")
    # 前向传播，得到最后结果
    train_image_feature = model(torch.tensor(train_image_feature))
    train_image_feature = train_image_feature.detach().numpy()
    # print(train_image_feature.shape)
    np.save("normal_feature/train_image_VGG_feature.npy", train_image_feature)
    # 前向传播，得到最后结果
    test_image_feature = model(torch.tensor(test_image_feature))
    test_image_feature = test_image_feature.detach().numpy()
    # print(test_image_feature.shape)
    np.save("normal_feature/test_image_VGG_feature.npy", test_image_feature)

if __name__ == "__main__":
    print("--------------一.利用Bert对文本特征进行提取---------------")
    extract_bert_features(model_name)
    print("\n--------------二.利用VGG对图像特征进行提取---------------")
    extract_vgg_features()
    