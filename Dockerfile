# 基于Python 3.9镜像（选择与原系统兼容的基础镜像）
FROM python:3.9-slim

# 复制依赖清单
COPY requirements.txt .

# 安装依赖
RUN pip install -r requirements.txt

# 复制项目代码
COPY . /app
WORKDIR /app