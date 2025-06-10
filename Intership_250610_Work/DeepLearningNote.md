
# 深度学习基础笔记

## 1. 深度学习训练基础概念

### 训练范式
- **训练过程本质**：两个循环（通常指训练循环和验证循环）
- **欠拟合(Underfitting)**：
  - 训练数据表现差
  - 验证数据表现差
- **过拟合(Overfitting)**：
  - 训练数据表现好
  - 验证数据表现差

---

## 2. 卷积神经网络（CNN）

### 2.1 卷积操作实现
```python
import torch
import torch.nn.functional as F

# 原始输入和卷积核
input = torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                      [0,1,0],
                      [2,1,0]])

# 调整维度（NCHW格式）
input = torch.reshape(input, (1,1,5,5))
kernel = torch.reshape(kernel, (1,1,3,3))

# 不同卷积方式
output1 = F.conv2d(input, kernel, stride=1)    # 步长1
output2 = F.conv2d(input, kernel, stride=2)    # 步长2 
output3 = F.conv2d(input, kernel, stride=1, padding=1)  # 填充1
```

### 2.2 图片卷积实战
```python
class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=0
        )
  
    def forward(self, x):
        return self.conv1(x)

# 可视化（TensorBoard）
writer = SummaryWriter("conv_logs")
model = CNN_Model()
for data in dataloader:
    imgs, _ = data
    outputs = model(imgs)
    # 调整维度可视化6通道输出
    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    writer.add_images("input", imgs, step)
    writer.add_images("output", outputs, step)
```

> **注**：启动TensorBoard：`tensorboard --logdir=conv_logs`

---

## 3. 池化层

### 3.1 最大池化实现
```python
class MaxPool_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(
            kernel_size=3,
            ceil_mode=False  # 是否向上取整
        )
  
    def forward(self, x):
        return self.maxpool(x)

# 数据可视化
writer = SummaryWriter("maxpool_logs")
model = MaxPool_Model()
for imgs, _ in dataloader:
    writer.add_images("input", imgs, step)
    writer.add_images("output", model(imgs), step)
```

### 池化类型
- 最大池化（Max Pooling）：取窗口内最大值
- 平均池化（Avg Pooling）：取窗口内平均值

---

## 关键注意事项
1. 输入数据需要转换为`torch.float32`类型（池化层不支持long类型）
2. 卷积/池化的输出尺寸计算公式：
   ```
   output_size = (input_size - kernel_size + 2*padding) / stride + 1
   ```
3. 使用TensorBoard时需注意：
   - 图像数据维度需符合`[N,C,H,W]`
   - 多通道输出需重组为可显示格式（如3的倍数通道）

## 4. 作业
###  搭建alexnet模型并进行训练
1. 作业保存在"QingHua_IntershipRepository\Intership_250610_Work\FundamentalsOfDeepLearning"路径下
2. alexnet.py为我的训练模型，MyTrain.py为我的训练文件
3. 结果保存至Intership_250610_Work\result中