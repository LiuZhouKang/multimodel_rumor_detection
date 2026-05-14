# 多模态大模型谣言检测

Multimodal Large Model Rumor Detection System

## 项目概述

本项目是一个基于多模态深度学习的谣言检测系统，结合了文本、图像和大语言模型推理等多种信息源来判断社交媒体上的信息是否为谣言。该系统能够处理中文微博和英文Twitter数据集，通过融合多种特征来提高谣言检测的准确性。

## 功能特点

- **多模态融合**：结合文本、图像和推理特征进行综合判断
- **大模型辅助**：利用GLM-4系列大语言模型进行文本和图像理解
- **双语支持**：支持中文微博和英文Twitter数据集
- **端到端流程**：从数据预处理到模型训练的完整pipeline

## 核心原理

本项目基于多模态机器学习理论，通过融合文本、图像和大模型推理三种不同模态的信息来提升谣言检测的准确性。系统的核心原理包括：

### 1. 多模态特征融合
- **文本特征**：使用BERT模型提取深层语义特征
- **图像特征**：使用VGG和CLIP模型提取视觉特征
- **推理特征**：利用大语言模型进行逻辑推理和上下文理解

### 2. 大模型辅助推理
- 利用GLM-4系列大模型对文本和图像进行深度理解
- 通过提示工程引导模型生成推理过程和判断依据
- 将推理结果转化为数值化特征向量

### 3. 跨模态关联分析
- 分析文本与图像之间的语义一致性
- 识别跨模态矛盾信息作为谣言线索
- 综合多维度证据进行最终判断

## 技术架构

系统包含以下几个核心模块：

1. **数据预处理模块** (`data_prepare.py`)：处理微博和Twitter数据集
2. **特征提取模块**：
   - CLIP模型提取图文特征
   - VGG模型提取图像特征
   - BERT模型提取文本特征
3. **大模型推理模块** (`judge_by_bigmodal.py`)：使用GLM-4进行推理分析
4. **特征融合模块** (`get_mixed_feature.py`)：融合多种特征向量
5. **分类模型** (`model_and_train.py`)：二分类器进行最终判断

## 工作流程

### 阶段一：数据准备与预处理
1. 加载微博或Twitter数据集
2. 解析文本、图像和标签信息
3. 数据清洗和格式标准化

### 阶段二：多模态特征提取
1. **CLIP特征提取**：提取图文联合特征
2. **VGG/BERT特征提取**：分别提取图像和文本基础特征
3. **大模型推理特征提取**：
   - 文本推理：分析文本内容的真实性
   - 图像推理：分析图像内容的真实性和相关性
   - 生成推理理由并转化为特征向量

### 阶段三：特征融合与建模
1. 将多种特征向量进行拼接融合
2. 构建1024维的综合特征向量
3. 使用三层神经网络进行谣言分类

### 阶段四：模型训练与评估
1. 使用交叉熵损失函数进行训练
2. 实时监控准确率、精确率、召回率和F1分数
3. 保存最优模型权重

## 主要创新点

### 1. 多模态大模型融合
- 首次将大语言模型的推理能力引入谣言检测任务
- 结合传统深度学习特征与大模型推理特征
- 实现了知识驱动与数据驱动的有机结合

### 2. 跨模态一致性检验
- 设计了文本-图像一致性检测机制
- 通过大模型分析跨模态信息的匹配度
- 有效识别故意误导性的图文组合

### 3. 推理可解释性
- 不仅输出检测结果，还提供推理过程
- 通过大模型生成判断依据，增强结果可信度
- 为谣言检测提供透明的决策路径

### 4. 双语支持架构
- 同时支持中文微博和英文Twitter数据
- 针对不同语言设计专门的提示模板
- 实现跨语言谣言检测能力

### 5. 自适应特征融合策略
- 动态调整不同模态特征的权重
- 根据数据特性优化特征融合方式
- 提高模型在不同场景下的泛化能力

## 依赖环境

- Python 3.8+
- PyTorch
- Transformers
- OpenAI API库
- ZhipuAI SDK
- NumPy, Pandas
- Scikit-learn
- 其他详见 `requirement.txt`

安装依赖：

```bash
pip install -r requirement.txt
```

## 数据集

项目支持两种数据集：

1. **微博数据集** (`MM17-WeiboRumorSet`)：中文社交媒体谣言数据
2. **Twitter数据集** (`image-verification-corpus`)：英文社交媒体验证数据

## 运行流程

完整的运行流程如下（参考 `bash.sh`）：

```bash
# 1. 图像修复（可选）
python image_repair.py --input image-verification-corpus/images --output repaired_images

# 2. 数据集预处理
python data_prepare.py --ratio 0.2 --data_from weibo

# 3. 提取CLIP特征
python clip_feature_process.py --data_from weibo

# 4. 提取VGG和BERT特征
python normal_feature_process.py --data_from weibo

# 5. 调用大模型API进行推理分析
python judge_by_bigmodal.py --data_from weibo

# 6. 处理推理结果获取特征
python reason_feature_process.py --data_from weibo

# 7. 特征融合
python get_mixed_feature.py --data_from weibo

# 8. 模型训练与测试
python model_and_train.py
```

## 参数说明

- `--data_from`: 数据集来源 ('weibo' 或 'Twitter')
- `--ratio`: 数据集采样比例 (默认 0.2)
- `--batch_size`: 批处理大小 (默认 32)
- `--num_workers`: 并行线程数 (默认 8)

## 模型结构

分类器采用三层全连接神经网络：

- 输入层：1024维融合特征
- 隐藏层1：256个神经元
- 隐藏层2：16个神经元
- 输出层：2个神经元（谣言/非谣言）

## 评估指标

模型输出以下评估指标：

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1 Score)

## 使用方法

1. 准备数据集：确保数据集文件夹结构正确
2. 配置API密钥：在 `judge_by_bigmodal.py` 中填入ZhipuAI API密钥
3. 根据需要调整参数
4. 按顺序执行上述运行流程

## 注意事项

1. 需要有效的ZhipuAI API密钥才能使用大模型推理功能
2. 确保数据集文件夹和保存路径存在
3. 根据硬件资源调整批处理大小和线程数
4. 大模型推理可能产生费用，请注意控制使用量

## 部署Web服务

本项目包含完整的Web前后端系统，提供可视化的谣言检测界面和RESTful API接口。

### 启动Web服务

```bash
# 启动Flask Web服务（默认端口5000）
python app.py
```

启动后访问 http://localhost:5000 即可使用Web界面。

### Web功能

- **可视化界面**：支持文本输入和图像上传，实时显示检测结果
- **进度展示**：WebSocket实时推送特征提取和推理进度
- **结果可视化**：概率分布图表、特征信息展示、推理说明
- **API接口**：提供完整的RESTful API供第三方调用

### API接口说明

1. **健康检查**: `GET /api/health`
2. **单条预测**: `POST /api/predict` (JSON格式)
3. **文件上传预测**: `POST /api/predict/upload` (Form格式)
4. **批量预测**: `POST /api/batch_predict` (最多10条)

### 系统架构

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   前端界面   │◄───►│ Flask后端服务 │◄───►│  深度学习模型  │
│ HTML/CSS/JS │     │ Flask+Socket │     │ BERT/VGG/CLIP│
└─────────────┘     └──────────────┘     └──────────────┘
                           │
                     ┌──────────────┐
                     │  Web界面功能  │
                     │ - 文本输入    │
                     │ - 图像上传    │
                     │ - 进度展示    │
                     │ - 结果可视化  │
                     └──────────────┘
```

## 文件结构

```
multimodel_rumor_detection/
├── app.py                     # Flask Web服务主程序
├── data_prepare.py            # 数据预处理
├── judge_by_bigmodal.py       # 大模型推理
├── model_and_train.py         # 模型训练
├── clip_feature_process.py    # CLIP特征提取
├── bert_feature_process.py    # BERT文本特征提取
├── vgg_feature_process.py     # VGG图像特征提取
── normal_feature_process.py  # 原始BERT/VGG联合提取(保留)
├── reason_feature_process.py  # 推理特征处理
├── get_mixed_feature.py       # 特征融合
├── image_repair.py            # 图像修复
├── bash.sh                    # 执行脚本
├── requirement.txt            # 依赖包
├── README.md                  # 项目说明
├── 部署指南.md                # Web服务部署文档
├── frontend/                  # Web前端文件
│   ├── templates/
│   │   ├── index.html        # 主页模板
│   │   └── train.html        # 训练页面模板
│   └── static/
│       ├── css/
│       │   ├── style.css     # 主页样式
│       │   └── train.css     # 训练页面样式
│       └── js/
│           ├── main.js       # 主页逻辑
│           └── train.js      # 训练页面逻辑
── uploads/                   # 上传文件临时目录
```

## 运行流程

完整的运行流程如下（参考 `bash.sh`）：

```bash
# 1. 图像修复（可选）
python image_repair.py --input image-verification-corpus/images --output repaired_images

# 2. 数据集预处理
python data_prepare.py --ratio 0.2 --data_from weibo

# 3. 提取CLIP特征
python clip_feature_process.py --data_from weibo

# 4. 提取BERT文本特征
python bert_feature_process.py --data_from weibo

# 5. 提取VGG图像特征
python vgg_feature_process.py --data_from weibo

# 6. 调用大模型API进行推理分析
python judge_by_bigmodal.py --data_from weibo

# 7. 处理推理结果获取特征
python reason_feature_process.py --data_from weibo

# 8. 特征融合
python get_mixed_feature.py --data_from weibo

# 9. 模型训练与测试
python model_and_train.py
```

### Web界面训练流程

通过Web界面可以可视化地完成整个训练流程：

1. 启动Web服务：`python app.py`
2. 访问训练页面：http://localhost:5000/train
3. 按步骤操作：
   - **数据准备**：选择数据集来源和采样比例
   - **特征提取**：依次提取CLIP、BERT、VGG特征
   - **模型训练**：设置训练参数并启动训练
   - **训练监控**：实时查看损失曲线和准确率

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

