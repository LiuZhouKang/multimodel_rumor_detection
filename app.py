#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态谣言检测系统 - 后端Flask服务
提供模型推理API和静态文件服务
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import base64
import tempfile
import logging
import subprocess
import re
import threading
import cv
from datetime import datetime
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# ==================== PyTorch 2.1.0与CLIP兼容性修复 ====================
# 修复 module 'torch.utils._pytree' has no attribute 'register_pytree_node' 错误
try:
    import torch.utils._pytree as pytree
    if not hasattr(pytree, 'register_pytree_node'):
        # 创建一个兼容的register_pytree_node函数
        def _compatible_register_pytree_node(cls, *args, **kwargs):
            """兼容PyTorch不同版本的register_pytree_node函数"""
            # 如果存在_register_pytree_node，尝试调用它
            if hasattr(pytree, '_register_pytree_node'):
                try:
                    # 尝试移除不兼容的参数
                    kwargs.pop('serialized_type_name', None)
                    kwargs.pop('flatten_fn', None)
                    kwargs.pop('unflatten_fn', None)
                    return pytree._register_pytree_node(cls, *args, **kwargs)
                except TypeError:
                    # 如果还是失败，尝试最少参数
                    try:
                        return pytree._register_pytree_node(cls)
                    except Exception:
                        pass
            # 静默失败，不抛出异常
            return None
        
        pytree.register_pytree_node = _compatible_register_pytree_node
except Exception:
    pass
# ======================================================================

# 导入项目原有模块
from model_and_train import BinaryClassifier

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask应用初始化
app = Flask(__name__, static_folder='frontend/static', template_folder='frontend/templates')
app.config['SECRET_KEY'] = 'rumor_detection_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 最大上传50MB
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_TEXT_LENGTH = 5000  # 最大文本长度
MODEL_PATH = 'best_model.pth'

# 创建上传目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局模型变量
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model = None
clip_preprocess = None
bert_model = None
bert_tokenizer = None
vgg_model = None
chinese_clip_model = None
chinese_clip_processor = None
bert_feature_reducer = None  # BERT特征维度缩减模型 (768->512)
vgg_feature_reducer = None   # VGG特征维度缩减模型

# 温度缩放参数（用于概率校准）
TEMPERATURE = 2.0  # 温度值，越大概率分布越平滑

# 大模型配置
LARGE_MODEL_API_KEY = None
LARGE_MODEL_PROVIDER = 'zhipu'  # 默认智谱


class BertFeatureReducer(nn.Module):
    """BERT特征维度缩减模型：768 -> 512"""
    def __init__(self):
        super(BertFeatureReducer, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x


class VGGFeatureReducer(nn.Module):
    """VGG特征维度缩减模型"""
    def __init__(self):
        super(VGGFeatureReducer, self).__init__()
        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        return x


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def load_models():
    """加载所有模型到内存"""
    global model, clip_model, clip_preprocess, bert_model, bert_tokenizer, vgg_model
    
    logger.info("开始加载模型...")
    
    try:
        # 加载分类器模型
        if os.path.exists(MODEL_PATH):
            model = BinaryClassifier(input_dim=1024)
            # 使用torch.load的兼容方式
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            logger.info(f"分类器模型加载成功: {MODEL_PATH}")
        else:
            logger.warning(f"未找到模型文件: {MODEL_PATH}，请先训练模型")
            model = None
        
        # 加载CLIP模型（增加错误处理）
        try:
            logger.info("加载CLIP模型...")
            import clip
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            clip_model.eval()
            logger.info("CLIP模型加载成功")
        except Exception as clip_error:
            logger.error(f"CLIP模型加载失败: {str(clip_error)}")
            logger.warning("将使用延迟加载模式，在首次推理时尝试加载CLIP模型")
            clip_model = None
        
        # 加载BERT模型（兼容PyTorch 2.1.0）
        try:
            logger.info("加载BERT模型...")
            from transformers import BertTokenizer, BertModel
            
            # 直接加载本地模型文件
            bert_tokenizer = BertTokenizer.from_pretrained(
                'corpus/chinese_L-12_H-768_A-12',
                local_files_only=True
            )
            
            # 使用from_pretrained但指定使用torch.load旧版本方式
            import transformers
            transformers.utils.import_utils.disable_progress_bar()
            
            bert_model = BertModel.from_pretrained(
                'corpus/chinese_L-12_H-768_A-12',
                local_files_only=True
            ).to(device)
            bert_model.eval()
            logger.info("BERT模型加载成功")
        except Exception as bert_error:
            logger.error(f"BERT模型加载失败: {str(bert_error)}")
            # 如果transformers版本不兼容，尝试手动加载
            try:
                logger.info("尝试手动加载BERT模型...")
                from transformers import BertConfig, BertModel, BertTokenizer
                
                # 加载配置
                config = BertConfig.from_pretrained(
                    'corpus/chinese_L-12_H-768_A-12',
                    local_files_only=True
                )
                
                # 创建模型
                bert_model = BertModel(config)
                
                # 使用torch.load加载权重
                state_dict = torch.load(
                    'corpus/chinese_L-12_H-768_A-12/pytorch_model.bin',
                    map_location=device
                )
                
                # 加载状态字典
                bert_model.load_state_dict(state_dict)
                bert_model.to(device)
                bert_model.eval()
                
                # 加载tokenizer
                bert_tokenizer = BertTokenizer.from_pretrained(
                    'corpus/chinese_L-12_H-768_A-12',
                    local_files_only=True
                )
                
                logger.info("BERT模型加载成功（手动加载）")
            except Exception as bert_error2:
                logger.error(f"BERT模型手动加载失败: {str(bert_error2)}")
                bert_model = None
                bert_tokenizer = None
        
        # 加载VGG模型
        try:
            logger.info("加载VGG模型...")
            import torchvision.models as models
            vgg_model = models.vgg19(pretrained=True).eval().to(device)
            logger.info("VGG模型加载成功")
        except Exception as vgg_error:
            logger.error(f"VGG模型加载失败: {str(vgg_error)}")
            vgg_model = None
        
        logger.info("模型加载完成！")
        return True
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return False


def extract_clip_features(text, image_path):
    """提取CLIP特征"""
    import clip
    
    global clip_model, clip_preprocess
    
    # 延迟加载CLIP模型（如果之前加载失败）
    if clip_model is None or clip_preprocess is None:
        logger.info("延迟加载CLIP模型...")
        try:
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            clip_model.eval()
            logger.info("CLIP模型延迟加载成功")
        except Exception as e:
            logger.error(f"CLIP模型延迟加载失败: {str(e)}")
            raise RuntimeError(f"CLIP模型加载失败: {str(e)}")
    
    # 提取文本特征
    text_inputs = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        text_feature = clip_model.encode_text(text_inputs).cpu().numpy()[0]
    
    # 提取图像特征
    image = Image.open(image_path).convert('RGB')
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = clip_model.encode_image(image_input).cpu().numpy()[0]
    
    return text_feature, image_feature


def extract_bert_features(text):
    """提取BERT特征"""
    global bert_model, bert_tokenizer, bert_feature_reducer
    
    # 延迟加载BERT模型（如果之前加载失败）
    if bert_model is None or bert_tokenizer is None:
        logger.info("延迟加载BERT模型...")
        try:
            from transformers import BertTokenizer, BertModel
            
            bert_tokenizer = BertTokenizer.from_pretrained(
                'corpus/chinese_L-12_H-768_A-12',
                local_files_only=True
            )
            
            bert_model = BertModel.from_pretrained(
                'corpus/chinese_L-12_H-768_A-12',
                local_files_only=True
            ).to(device)
            bert_model.eval()
            logger.info("BERT模型延迟加载成功")
        except Exception as e:
            logger.error(f"BERT模型延迟加载失败: {str(e)}")
            raise RuntimeError(f"BERT模型加载失败: {str(e)}")
    
    # 延迟加载BERT特征缩减模型
    if bert_feature_reducer is None:
        bert_feature_reducer = BertFeatureReducer().to(device)
        bert_feature_reducer.eval()
        logger.info("BERT特征缩减模型已初始化")
    
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding='max_length'
    ).to(device)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        text_feature = outputs.last_hidden_state[:, 0, :]  # 768维
        # 维度缩减：768 -> 512
        text_feature = bert_feature_reducer(text_feature)
        text_feature = text_feature.cpu().numpy()[0]
    
    return text_feature


def extract_vgg_features(image_path):
    """提取VGG特征"""
    from torchvision import transforms
    
    global vgg_model, vgg_feature_reducer
    
    # 延迟加载VGG模型（如果之前加载失败）
    if vgg_model is None:
        logger.info("延迟加载VGG模型...")
        try:
            import torchvision.models as models
            vgg_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).eval().to(device)
            logger.info("VGG模型延迟加载成功")
        except Exception as e:
            logger.error(f"VGG模型延迟加载失败: {str(e)}")
            raise RuntimeError(f"VGG模型加载失败: {str(e)}")
    
    # 延迟加载VGG特征缩减模型
    if vgg_feature_reducer is None:
        vgg_feature_reducer = VGGFeatureReducer().to(device)
        vgg_feature_reducer.eval()
        logger.info("VGG特征缩减模型已初始化")
    
    # VGG预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载和处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 提取特征（使用第6层）
    feature_extractor = torch.nn.Sequential(*list(vgg_model.features.children())[:6])
    with torch.no_grad():
        features = feature_extractor(image_tensor)  # 保持为tensor
        # 维度缩减 -> 512
        features = vgg_feature_reducer(features)
        features = features.cpu().numpy()[0]
    
    return features


def call_large_model_for_reason(text, image_path):
    """调用大模型对图文进行推理判断"""
    global LARGE_MODEL_API_KEY, LARGE_MODEL_PROVIDER
    
    try:
        # 对文本进行推理
        text_messages = [
            {"role": "system", "content": "给定以下新闻，请判断其真实性（真或假），并简要说明判断的原因。请避免提供模棱两可的评估。"},
            {"role": "user", "content": text}
        ]
        
        # 对图片进行推理
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        image_messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": "这张图片来自一篇新闻报道，请根据图片内容判断其真实性，并简要说明理由。请避免提供图片内容需要进一步核实、模棱两可的评估。"}
            ]
        }]
        
        text_reason = ""
        image_reason = ""
        
        # 调用智谱AI
        if LARGE_MODEL_PROVIDER == 'zhipu':
            from zhipuai import ZhipuAI
            api_key = LARGE_MODEL_API_KEY or '33b4969db86e4599a8b47a61e2125a0a.WtHV4Q6OjGKXbytd'
            logger.info(f"使用智谱AI进行推理，API Key: {'***' + api_key[-4:] if api_key else 'None'}")
            
            try:
                client = ZhipuAI(api_key=api_key)
                
                text_response = client.chat.completions.create(
                    model="glm-4-flash",
                    messages=text_messages,
                    max_tokens=150
                )
                if text_response and hasattr(text_response, 'choices') and text_response.choices:
                    text_reason = text_response.choices[0].message.content
                else:
                    logger.warning("智谱AI文本推理返回为空")
            except Exception as e:
                logger.error(f"智谱AI文本推理失败: {str(e)}")
            
            try:
                image_response = client.chat.completions.create(
                    model="glm-4v-flash",
                    messages=image_messages,
                    max_tokens=150
                )
                if image_response and hasattr(image_response, 'choices') and image_response.choices:
                    image_reason = image_response.choices[0].message.content
                else:
                    logger.warning("智谱AI图像推理返回为空")
            except Exception as e:
                logger.error(f"智谱AI图像推理失败: {str(e)}")
        
        # 调用通义千问
        elif LARGE_MODEL_PROVIDER == 'qwen':
            import dashscope
            from dashscope import Generation, MultiModalConversation
            api_key = LARGE_MODEL_API_KEY or 'sk-4ba3868d640749f19300ff010ff50a42'
            dashscope.api_key = api_key
            
            try:
                text_response = Generation.call(
                    model='qwen-turbo',
                    messages=text_messages,
                    result_format='message',
                )
                if text_response.status_code == 200 and text_response.output:
                    content = text_response.output.choices[0].message.content if text_response.output.choices else None
                    if content:
                        text_reason = content[0].get('text', '') if isinstance(content, list) else content
                else:
                    logger.warning(f"千问文本推理失败: {text_response.code} - {text_response.message}")
            except Exception as e:
                logger.error(f"千问文本推理异常: {str(e)}")
            
            try:
                image_response = MultiModalConversation.call(
                    model='qwen-vl-max',
                    messages=image_messages,
                )
                if image_response.status_code == 200 and image_response.output:
                    content = image_response.output.choices[0].message.content if image_response.output.choices else None
                    if content:
                        image_reason = content[0].get('text', '') if isinstance(content, list) else content
                else:
                    logger.warning(f"千问图像推理失败: {image_response.code} - {image_response.message}")
            except Exception as e:
                logger.error(f"千问图像推理异常: {str(e)}")
        
        # 如果推理结果为空，使用默认值
        if not text_reason:
            text_reason = "该内容真实性需要进一步核实。"
        if not image_reason:
            image_reason = "该图片内容需要进一步核实。"
        
        return text_reason, image_reason
        
    except Exception as e:
        logger.error(f"大模型推理失败: {str(e)}")
        # 返回默认推理内容
        default_reason = "无法判断其真实性"
        return default_reason, default_reason


def extract_reason_features_with_chinese_clip(text_reason, image_reason):
    """使用Chinese CLIP提取推理文本的特征"""
    global chinese_clip_model, chinese_clip_processor
    
    # 延迟加载Chinese CLIP模型
    if chinese_clip_model is None or chinese_clip_processor is None:
        logger.info("延迟加载Chinese CLIP模型...")
        try:
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
            model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
            cache_dir = "corpus/chinese-clip-cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            chinese_clip_model = ChineseCLIPModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
            chinese_clip_processor = ChineseCLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            chinese_clip_model.eval()
            logger.info("Chinese CLIP模型延迟加载成功")
        except Exception as e:
            logger.error(f"Chinese CLIP模型延迟加载失败: {str(e)}")
            raise RuntimeError(f"Chinese CLIP模型加载失败: {str(e)}")
    
    with torch.no_grad():
        # 提取文本推理特征
        text_inputs = chinese_clip_processor(text=[text_reason], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        text_features = chinese_clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_feature = text_features.cpu().numpy()[0]
        
        # 提取图像推理特征
        image_inputs = chinese_clip_processor(text=[image_reason], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        image_features = chinese_clip_model.get_text_features(**image_inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        image_feature = image_features.cpu().numpy()[0]
    
    return text_feature, image_feature


def calculate_cosine_similarity(v1, v2):
    """计算余弦相似度"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return abs(dot_product / (norm_v1 * norm_v2))


def enhance_image(image_path):
    """
    图像质量增强（调用image_repair.py脚本）
    """
    try:
        from tempfile import NamedTemporaryFile
        import subprocess
        import os
        
        # 创建临时输出目录
        temp_dir = os.path.dirname(image_path)
        output_dir = os.path.join(temp_dir, "temp_enhanced")
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件路径 - 使用固定命名规则
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        enhanced_path = os.path.join(output_dir, f"{name}_enhanced.jpg")  # 统一输出为jpg格式
        
        # 调用image_repair.py进行增强
        result = subprocess.run([
            sys.executable, 'image_repair.py',
            '--input', image_path,
            '--output', output_dir,
            '--mode', 'enhance'  # 使用增强模式
        ], capture_output=True, text=True, timeout=30)
        
        # 检查增强后的文件是否存在
        if os.path.exists(enhanced_path):
            logger.info(f"图像增强完成: {image_path} -> {enhanced_path}")
            return enhanced_path
        else:
            # 如果增强失败，查找输出目录中最新的文件
            files = []
            for file in os.listdir(output_dir):
                if file.startswith(name) and file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    filepath = os.path.join(output_dir, file)
                    mtime = os.path.getmtime(filepath)
                    files.append((mtime, filepath))
            
            if files:
                # 返回最新修改的文件
                files.sort(reverse=True)
                latest_file = files[0][1]
                logger.info(f"图像增强完成 (找到文件): {image_path} -> {latest_file}")
                return latest_file
            else:
                logger.warning(f"图像增强失败，使用原图: {image_path}")
                return image_path
                
    except subprocess.TimeoutExpired:
        logger.error("图像增强超时，使用原图")
        return image_path
    except Exception as e:
        logger.error(f"图像增强失败: {str(e)}")
        return image_path  # 失败时返回原图


def predict_rumor(text, image_path, enable_image_enhance=True):
    """
    完整的谣言检测流程
    返回: (prediction, confidence, reasoning)
    """
    try:
        socketio.emit('progress', {'step': '开始处理图像...', 'percent': 5})
        
        # 0. 图像质量增强（与训练时保持一致）
        original_image_path = image_path
        enhanced_image_path = image_path
        if enable_image_enhance:
            enhanced_image_path = enhance_image(image_path)
            image_path = enhanced_image_path
            socketio.emit('progress', {'step': '图像增强完成', 'percent': 10})
        else:
            socketio.emit('progress', {'step': '跳过图像增强', 'percent': 10})
        
        # 1. 提取CLIP特征
        clip_text, clip_image = extract_clip_features(text, image_path)
        socketio.emit('progress', {'step': 'CLIP特征提取完成', 'percent': 25})
        
        # 2. 提取BERT特征
        bert_text = extract_bert_features(text)
        socketio.emit('progress', {'step': 'BERT特征提取完成', 'percent': 40})
        
        # 3. 提取VGG特征
        vgg_image = extract_vgg_features(image_path)
        socketio.emit('progress', {'step': 'VGG特征提取完成', 'percent': 55})
        
        # 4. 调用大模型进行推理并提取推理特征
        socketio.emit('progress', {'step': '调用大模型进行推理...', 'percent': 65})
        text_reason, image_reason = call_large_model_for_reason(text, image_path)
        socketio.emit('progress', {'step': '提取推理特征...', 'percent': 80})
        reason_text_feature, reason_image_feature = extract_reason_features_with_chinese_clip(text_reason, image_reason)
        socketio.emit('progress', {'step': '推理特征提取完成', 'percent': 90})
        
        # 5. 计算相似度分数（与训练时保持一致）
        sim_text = calculate_cosine_similarity(bert_text, clip_text)
        sim_image = calculate_cosine_similarity(vgg_image, clip_image)
        sim_cross1 = calculate_cosine_similarity(vgg_image, bert_text)
        sim_cross2 = calculate_cosine_similarity(clip_image, bert_text)
        sim_cross3 = calculate_cosine_similarity(clip_text, vgg_image)
        simulation_score = (sim_text + sim_image + sim_cross1 + sim_cross2 + sim_cross3) / 5
        
        # 6. 拼接特征（与训练时的 get_mixed_feature.py 保持一致）
        # mixed_1: 拼接BERT文本特征和VGG图像特征
        mixed_1 = np.concatenate([bert_text, vgg_image])
        
        # mixed_2: 拼接推理文本特征和推理图像特征
        mixed_2 = np.concatenate([reason_text_feature, reason_image_feature])
        
        # 7. 加权融合两个表征
        mixed_feature = (1 - simulation_score) * mixed_1 + simulation_score * mixed_2
        
        # 8. 模型预测
        if model is None:
            return {
                'error': '模型未加载，请先训练模型',
                'prediction': None,
                'confidence': 0
            }
        
        input_tensor = torch.FloatTensor(mixed_feature).unsqueeze(0).to(device)
        logger.info(f"输入特征维度: {mixed_feature.shape}")
        
        with torch.no_grad():
            output = model(input_tensor)
            logger.info(f"模型原始输出 (logits): {output}")
            
            # 应用温度缩放进行概率校准
            calibrated_output = output / TEMPERATURE
            logger.info(f"温度缩放后输出: {calibrated_output}")
            
            probabilities = torch.softmax(calibrated_output, dim=1).cpu().numpy()[0]
            logger.info(f"Softmax后概率: {probabilities}")
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]
            logger.info(f"预测结果: {prediction}, 置信度: {confidence:.4f}")
        
        socketio.emit('progress', {'step': '预测完成', 'percent': 100})
        
        # 9. 生成推理说明
        if prediction == 1:
            result_text = "谣言"
            reasoning = f"系统分析结果表明该内容为谣言的可能性较高（置信度: {confidence:.2%}）。大模型推理：{text_reason}"
        else:
            result_text = "非谣言"
            reasoning = f"系统分析结果表明该内容为非谣言的可能性较高（置信度: {confidence:.2%}）。大模型推理：{text_reason}"
        
        result = {
            'prediction': result_text,
            'confidence': float(confidence),
            'probabilities': {
                '非谣言': float(probabilities[0]),
                '谣言': float(probabilities[1])
            },
            'simulation_score': float(simulation_score),
            'reasoning': reasoning,
            'large_model_reasoning': {
                'text': text_reason,
                'image': image_reason
            },
            'feature_stats': {
                'bert_dim': len(bert_text),
                'vgg_dim': len(vgg_image),
                'reason_text_dim': len(reason_text_feature),
                'reason_image_dim': len(reason_image_feature),
                'mixed_dim': len(mixed_feature)
            }
        }
        
        # 清理临时增强图像
        if enhanced_image_path != original_image_path and os.path.exists(enhanced_image_path):
            try:
                os.remove(enhanced_image_path)
            except:
                pass
        
        return result
        
    except Exception as e:
        logger.error(f"预测过程出错: {str(e)}")
        return {
            'error': str(e),
            'prediction': None,
            'confidence': 0
        }


# ==================== 路由定义 ====================

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')


@app.route('/train')
def train_page():
    """渲染训练页面"""
    return render_template('train.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    预测接口
    接收: JSON {text: str, image: base64_encoded_string, model_provider: str, api_key: str}
    返回: JSON {prediction, confidence, probabilities, reasoning}
    """
    global LARGE_MODEL_API_KEY, LARGE_MODEL_PROVIDER
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
        
        text = data.get('text', '').strip()
        image_data = data.get('image', '')
        
        # 获取大模型配置
        LARGE_MODEL_PROVIDER = data.get('model_provider', 'zhipu')
        LARGE_MODEL_API_KEY = data.get('api_key', None)
        
        # 获取图像增强配置
        enable_image_enhance = data.get('enable_image_enhance', True)
        
        if not text:
            return jsonify({'error': '文本内容不能为空'}), 400
        
        if not image_data:
            return jsonify({'error': '图像数据不能为空'}), 400
        
        # 检查文本长度
        if len(text) > ALLOWED_TEXT_LENGTH:
            return jsonify({'error': f'文本长度超过限制（最大{ALLOWED_TEXT_LENGTH}字符）'}), 400
        
        # 解码base64图像
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({'error': f'图像数据格式错误: {str(e)}'}), 400
        
        # 保存临时图像文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=UPLOAD_FOLDER) as tmp:
            tmp.write(image_bytes)
            image_path = tmp.name
        
        try:
            # 执行预测
            result = predict_rumor(text, image_path, enable_image_enhance)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            return jsonify({
                'success': True,
                'data': result
            })
            
        finally:
            # 清理临时文件
            if os.path.exists(image_path):
                os.remove(image_path)
                
    except Exception as e:
        logger.error(f"预测接口异常: {str(e)}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


@app.route('/api/predict/upload', methods=['POST'])
def predict_with_upload():
    """
    预测接口（文件上传方式）
    接收: Form {text: str, image: File}
    返回: JSON {prediction, confidence, probabilities, reasoning}
    """
    try:
        text = request.form.get('text', '').strip()
        
        if not text:
            return jsonify({'error': '文本内容不能为空'}), 400
        
        if len(text) > ALLOWED_TEXT_LENGTH:
            return jsonify({'error': f'文本长度超过限制（最大{ALLOWED_TEXT_LENGTH}字符）'}), 400
        
        if 'image' not in request.files:
            return jsonify({'error': '未找到上传的图像文件'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': '未选择图像文件'}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({'error': '不支持的图像格式'}), 400
        
        # 保存上传的图像
        filename = secure_filename(f"{datetime.now().timestamp()}_{image_file.filename}")
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)
        
        try:
            # 执行预测
            result = predict_rumor(text, image_path)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            return jsonify({
                'success': True,
                'data': result
            })
            
        finally:
            # 清理临时文件
            if os.path.exists(image_path):
                os.remove(image_path)
                
    except Exception as e:
        logger.error(f"上传预测接口异常: {str(e)}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    批量预测接口
    接收: JSON {items: [{text, image}, ...]}
    返回: JSON {results: [...]}
    """
    try:
        data = request.get_json()
        
        if not data or 'items' not in data:
            return jsonify({'error': '请求数据格式错误'}), 400
        
        items = data['items']
        
        if len(items) > 10:
            return jsonify({'error': '批量预测最多支持10条数据'}), 400
        
        results = []
        for i, item in enumerate(items):
            text = item.get('text', '').strip()
            image_data = item.get('image', '')
            
            if not text or not image_data:
                results.append({
                    'index': i,
                    'error': '文本或图像数据缺失'
                })
                continue
            
            # 解码图像
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=UPLOAD_FOLDER) as tmp:
                tmp.write(image_bytes)
                image_path = tmp.name
            
            try:
                result = predict_rumor(text, image_path)
                result['index'] = i
                results.append(result)
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)
        
        return jsonify({
            'success': True,
            'data': {
                'total': len(items),
                'results': results
            }
        })
        
    except Exception as e:
        logger.error(f"批量预测接口异常: {str(e)}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


# WebSocket事件
@socketio.on('connect')
def handle_connect():
    """客户端连接事件"""
    logger.info("客户端已连接")
    emit('connected', {'message': '已成功连接到服务器'})


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开事件"""
    logger.info("客户端已断开连接")


# ==================== 训练API接口 ====================

@app.route('/api/train/prepare_data', methods=['POST'])
def api_prepare_data():
    """准备数据集接口"""
    try:
        data = request.get_json()
        data_from = data.get('data_from', 'weibo')
        ratio = data.get('ratio', 0.2)
        enable_image_enhance = data.get('enable_image_enhance', True)
        
        # 调用数据预处理模块
        import subprocess
        
        # 构建命令参数
        cmd_args = [
            'python', 'data_prepare.py',
            '--ratio', str(ratio),
            '--data_from', data_from
        ]
        
        if enable_image_enhance:
            cmd_args.append('--enhance_images')
        else:
            cmd_args.append('--no_enhance_images')
        
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300
        )
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': f'{data_from}数据集准备完成 (图像增强: {enable_image_enhance})'
            })
        else:
            return jsonify({
                'error': f'数据准备失败: {result.stderr}'
            }), 500
            
    except Exception as e:
        logger.error(f"数据准备失败: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/extract_clip', methods=['POST'])
def api_extract_clip():
    """提取CLIP特征接口"""
    try:
        data = request.get_json()
        data_from = data.get('data_from', 'weibo')
        
        import subprocess
        result = subprocess.run(
            ['python', 'clip_feature_process.py', '--data_from', data_from],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=1000
        )
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'CLIP特征提取完成'})
        else:
            return jsonify({'error': f'CLIP特征提取失败: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/extract_bert', methods=['POST'])
def api_extract_bert():
    """提取BERT特征接口"""
    try:
        data = request.get_json()
        data_from = data.get('data_from', 'weibo')
        
        import subprocess
        result = subprocess.run(
            ['python', 'bert_feature_process.py', '--data_from', data_from],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=1000
        )
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'BERT特征提取完成'})
        else:
            return jsonify({'error': f'BERT特征提取失败: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/extract_vgg', methods=['POST'])
def api_extract_vgg():
    """提取VGG特征接口"""
    try:
        data = request.get_json()
        data_from = data.get('data_from', 'weibo')
        
        import subprocess
        result = subprocess.run(
            ['python', 'vgg_feature_process.py', '--data_from', data_from],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=1000
        )
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'VGG特征提取完成'})
        else:
            return jsonify({'error': f'VGG特征提取失败: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/extract_fusion', methods=['POST'])
def api_extract_fusion():
    """特征融合接口"""
    try:
        data = request.get_json()
        data_from = data.get('data_from', 'weibo')
        
        import subprocess
        result = subprocess.run(
            ['python', 'get_mixed_feature.py', '--data_from', data_from],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=1000
        )
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': '特征融合完成'})
        else:
            return jsonify({'error': f'特征融合失败: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/extract_reason', methods=['POST'])
def api_extract_reason():
    """提取推理特征接口（异步执行）"""
    import threading
    
    try:
        data = request.get_json()
        data_from = data.get('data_from', 'weibo')
        api_key = data.get('api_key')
        model_provider = data.get('model_provider', 'zhipu')  # 默认智谱
        
        # 构建命令行参数（使用参数传递API Key，避免修改源文件触发服务器重启）
        cmd_args = [
            'python', 'judge_by_bigmodal.py',
            '--data_from', data_from,
            '--model_provider', model_provider
        ]
        
        # 如果提供了API Key，通过参数传递
        if api_key:
            cmd_args.extend(['--api_key', api_key])
        
        # 在后台线程中异步执行
        def run_extraction():
            try:
                provider_name = '智谱GLM' if model_provider == 'zhipu' else '通义千问'
                socketio.emit('training_log', {'level': 'info', 'message': f'开始调用{provider_name}进行推理分析...'})
                socketio.emit('training_progress', {'stage': 'reason', 'status': 'running', 'message': '推理中...'})
                
                # 执行大模型推理
                result = subprocess.run(
                    cmd_args,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=3600  # 1小时超时
                )
                
                if result.returncode == 0:
                    socketio.emit('training_log', {'level': 'success', 'message': '大模型推理完成，正在处理特征...'})
                    socketio.emit('training_progress', {'stage': 'reason', 'status': 'running', 'message': '处理特征...'})
                    
                    # 处理推理特征
                    result2 = subprocess.run(
                        ['python', 'reason_feature_process.py', '--data_from', data_from],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=600
                    )
                    
                    if result2.returncode == 0:
                        socketio.emit('training_log', {'level': 'success', 'message': '推理特征提取完成！'})
                        socketio.emit('training_progress', {'stage': 'reason', 'status': 'completed', 'message': '完成'})
                    else:
                        socketio.emit('training_log', {'level': 'error', 'message': f'推理特征处理失败: {result2.stderr}'})
                        socketio.emit('training_progress', {'stage': 'reason', 'status': 'error', 'message': '失败'})
                else:
                    socketio.emit('training_log', {'level': 'error', 'message': f'大模型推理失败: {result.stderr}'})
                    socketio.emit('training_progress', {'stage': 'reason', 'status': 'error', 'message': '失败'})
                    
            except Exception as e:
                socketio.emit('training_log', {'level': 'error', 'message': f'推理特征提取异常: {str(e)}'})
                socketio.emit('training_progress', {'stage': 'reason', 'status': 'error', 'message': str(e)})
        
        # 启动后台线程
        thread = threading.Thread(target=run_extraction, daemon=True)
        thread.start()
        
        # 立即返回，不阻塞前端
        return jsonify({
            'success': True, 
            'message': '大模型推理任务已启动，请在日志中查看进度'
        })
        
    except Exception as e:
        logger.error(f"推理特征提取接口异常: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/start', methods=['POST'])
def api_start_training():
    """开始训练接口（异步执行）"""
    import threading
    
    try:
        data = request.get_json()
        data_from = data.get('data_from', 'weibo')
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 32)
        learning_rate = data.get('learning_rate', 0.001)
        dropout = data.get('dropout', 0.2)
        
        # 在后台线程中异步执行
        def run_training():
            try:
                # 执行特征融合
                socketio.emit('training_log', {'level': 'info', 'message': f'开始特征融合...'})
                socketio.emit('training_progress', {'stage': 'fusion', 'status': 'running', 'message': '融合中...'})
                
                result = subprocess.run(
                    ['python', 'get_mixed_feature.py', '--data_from', data_from],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=1000
                )
                
                if result.returncode != 0:
                    socketio.emit('training_log', {'level': 'error', 'message': f'特征融合失败: {result.stderr}'})
                    socketio.emit('training_progress', {'stage': 'fusion', 'status': 'error', 'message': '失败'})
                    return
                
                socketio.emit('training_log', {'level': 'success', 'message': '特征融合完成'})
                socketio.emit('training_progress', {'stage': 'fusion', 'status': 'completed', 'message': '完成'})
                
                # 开始训练
                socketio.emit('training_log', {'level': 'info', 'message': f'开始训练模型 (epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, dropout={dropout})...'})
                socketio.emit('training_progress', {'stage': 'training', 'status': 'running', 'message': '训练中...'})
                
                # 使用命令行参数传递训练配置，避免修改文件触发Flask自动重载
                train_cmd = [
                    'python', 'model_and_train.py',
                    '--epochs', str(epochs),
                    '--lr', str(learning_rate),
                    '--batch_size', str(batch_size),
                    '--dropout', str(dropout)
                ]
                
                # 执行训练并实时获取输出
                process = subprocess.Popen(
                    train_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    universal_newlines=True
                )
                
                # 实时读取训练日志
                best_metrics_data = {}
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        # 解析训练输出
                        if 'Epoch' in line and 'Loss' in line:
                            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                            loss_match = re.search(r'Loss: ([\d.]+)', line)
                            acc_match = re.search(r'Test Accuracy: ([\d.]+)', line)
                            
                            if epoch_match and loss_match and acc_match:
                                epoch = int(epoch_match.group(1))
                                total = int(epoch_match.group(2))
                                loss = float(loss_match.group(1))
                                accuracy = float(acc_match.group(1))
                                
                                socketio.emit('training_log', {'level': 'info', 'message': line})
                                socketio.emit('training_progress', {
                                    'stage': 'training',
                                    'epoch': epoch,
                                    'total_epochs': total,
                                    'loss': loss,
                                    'accuracy': accuracy
                                })
                                socketio.emit('training_metrics', {
                                    'epoch': epoch,
                                    'loss': loss,
                                    'accuracy': accuracy
                                })
                        elif 'Best Performance' in line:
                            socketio.emit('training_log', {'level': 'success', 'message': line})
                        elif 'Accuracy:' in line:
                            acc_val_match = re.search(r'Accuracy: ([\d.]+)%', line)
                            if acc_val_match:
                                best_metrics_data['best_accuracy'] = float(acc_val_match.group(1)) / 100
                                socketio.emit('training_log', {'level': 'success', 'message': line})
                        elif 'Precision:' in line:
                            prec_match = re.search(r'Precision: ([\d.]+)', line)
                            if prec_match:
                                best_metrics_data['best_precision'] = float(prec_match.group(1))
                                socketio.emit('training_log', {'level': 'success', 'message': line})
                        elif 'Recall:' in line:
                            recall_match = re.search(r'Recall: ([\d.]+)', line)
                            if recall_match:
                                best_metrics_data['best_recall'] = float(recall_match.group(1))
                                socketio.emit('training_log', {'level': 'success', 'message': line})
                        elif 'F1 Score:' in line:
                            f1_match = re.search(r'F1 Score: ([\d.]+)', line)
                            if f1_match:
                                best_metrics_data['best_f1'] = float(f1_match.group(1))
                                socketio.emit('training_log', {'level': 'success', 'message': line})
                        elif 'Best Performance at Epoch' in line:
                            epoch_match = re.search(r'Best Performance at Epoch (\d+)', line)
                            if epoch_match:
                                best_metrics_data['best_epoch'] = int(epoch_match.group(1))
                
                process.wait()
                
                if process.returncode == 0:
                    socketio.emit('training_log', {'level': 'success', 'message': '训练完成！模型已保存为best_model.pth'})
                    socketio.emit('training_progress', {'stage': 'training', 'status': 'completed', 'message': '完成'})
                    
                    # 发送最佳性能指标到前端
                    if best_metrics_data:
                        socketio.emit('training_metrics', best_metrics_data)
                    
                    # 重新加载模型
                    load_models()
                else:
                    socketio.emit('training_log', {'level': 'error', 'message': '训练失败'})
                    socketio.emit('training_progress', {'stage': 'training', 'status': 'error', 'message': '失败'})
                    
            except Exception as e:
                socketio.emit('training_log', {'level': 'error', 'message': f'训练异常: {str(e)}'})
                socketio.emit('training_progress', {'stage': 'training', 'status': 'error', 'message': str(e)})
        
        # 启动后台线程
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        
        # 立即返回，不阻塞前端
        return jsonify({
            'success': True, 
            'message': '训练任务已启动，请在日志中查看进度'
        })
        
    except Exception as e:
        logger.error(f"训练接口异常: {str(e)}")
        socketio.emit('training_log', {'level': 'error', 'message': f'训练失败: {str(e)}'})
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/export_model', methods=['GET'])
def api_export_model():
    """导出模型接口"""
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': '模型文件不存在'}), 404
        
        return send_from_directory(
            os.path.dirname(os.path.abspath(MODEL_PATH)),
            os.path.basename(MODEL_PATH),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '请求的资源不存在'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': '请求方法不允许'}), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500


# ==================== 主程序入口 ====================

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("多模态谣言检测系统启动")
    logger.info("=" * 50)
    
    # 加载模型
    if load_models():
        logger.info("系统初始化成功，准备接收请求")
    else:
        logger.warning("模型加载失败，部分功能可能不可用")
    
    # 启动服务
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
