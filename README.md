# 实习项目仓库 (QingHua_IntershipRepository)

一个用于记录和提交实习期间工作内容的仓库，以及相关学习笔记。

## 📁 项目结构

### 🗂️ 主要目录说明

#### 📅 `Intership_250609_Work/`
**日期：2024年6月9日工作内容**
- `multispectral_to_rgb.py` - 多光谱图像转RGB图像的处理脚本
- `README.md` - 当日详细的工作笔记和学习记录
- `test1.py` ~ `test9.py` - 各种实验和测试脚本，用于验证不同的算法和功能

#### 📅 `Intership_250610_Work/`
**日期：2024年6月10日工作内容**
- `README.md` - 深度学习相关的学习笔记
- `FundamentalsOfDeepLearning/` - 深度学习基础实践代码
  - `alexnet.py` - AlexNet网络架构实现
  - `conv2d.py` / `conv2d_2.py` - 卷积层实现和测试
  - `PoolingLayer.py` - 池化层实现
  - `model.py` - 模型定义文件
  - `MyTrain.py` - 自定义训练脚本
  - `train_GPU_1.py` / `train_GPU_2.py` - GPU训练脚本（不同版本）
  - `test.py` - 功能测试脚本

#### 📅 `Intership_250611_Work/`
**日期：2024年6月11日工作内容**
- `dataset.py` - 自定义数据集类实现，用于图像分类任务的数据加载
- `ActivateFunction.py` - 激活函数实验脚本，包含ReLU和Sigmoid函数的可视化
- `datasets/` - 数据集处理相关文件
  - `train.txt` / `val.txt` - 训练集和验证集的标签文件
  - `deal_with_datasets.py` - 数据集预处理脚本
  - `train_test_spilt.py` - 训练测试集分割脚本
  - `train_output/` - 按类别组织的图像数据集目录

#### 📅 `Intership_250612_Work/`
**日期：2024年6月12日工作内容**
- `vit_1d.py` - Vision Transformer (ViT) 模型实现，包含注意力机制和Transformer架构
- `Mytrain_vit.py` - ViT模型训练脚本
- `dataset.py` - 改进的数据集类，增加了错误处理和异常情况处理

#### 📅 `Intership_250615Day7_Work/`
**日期：2024年6月15日工作内容**
- `yolov8.png` - YOLOv8算法网络结构流程图
- `README.md` - YOLOv8算法详细解释和流程分析，包含网络各模块功能说明

#### 📊 `result/`
**输出结果存储目录**
- `ResultOf6_10/` - 6月10日的实验结果
  - `ResultOfTrain.png` - 训练过结果

## 🔧 技术栈

- **Python** - 主要编程语言
- **PyTorch** - 深度学习框架，用于神经网络实现和训练
- **图像处理** - 多光谱图像处理和转换
- **数据可视化** - TensorBoard可视化，训练结果和数据分析
- **数据集处理** - 图像分类数据集的组织和预处理
- **Transformer架构** - Vision Transformer (ViT) 模型实现
- **目标检测算法** - YOLOv8算法流程分析和网络结构研究

## 📝 工作内容概述

### 第一天 (6/9)
- 多光谱图像处理算法研究与实现
- 相关测试脚本开发和验证
- 详细工作笔记记录

### 第二天 (6/10)  
- 深度学习基础理论学习
- 卷积神经网络（CNN）实现
- AlexNet架构复现
- GPU加速训练实验

### 第三天 (6/11)
- 图像分类数据集处理和组织
- 自定义PyTorch数据集类开发
- 激活函数（ReLU、Sigmoid）的实验
- 数据集训练测试分割实现

### 第四天 (6/12)
- Vision Transformer (ViT) 模型架构实现
- 自注意力机制和Transformer模块开发
- 改进的数据集类
- ViT模型训练和评估流程实现

### 第七天 (6/15)
- 绘制YOLOv8网络结构流程图
- 目标检测算法原理和流程理解

**备注**：本仓库持续更新中，记录实习期间的学习成果和技术探索。
