
import time
import os

import torch.optim
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import ImageTxtDataset
from vit_1d import ViT

train_data = ImageTxtDataset(r"E:\QingHua_IntershipRepository\Project_classStudy\Intership_250611_work\dataset\train.txt", r"E:\QingHua_IntershipRepository\Project_classStudy\Intership_250611_work\dataset\train_output",
                             transforms.Compose([
                                 transforms.Resize((256, 256)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.299, 0.224, 0.225])]))

test_data = ImageTxtDataset(r"E:\QingHua_IntershipRepository\Project_classStudy\Intership_250611_work\dataset\val.txt", r"E:\QingHua_IntershipRepository\Project_classStudy\Intership_250611_work\dataset\train_val",
                             transforms.Compose([
                                 transforms.Resize((256, 256)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.299, 0.224, 0.225])]))

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度{train_data_size}")
print(f"测试数据集的长度{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data,batch_size=16)
test_loader = DataLoader(test_data,batch_size=16)

# 创建网络模型
vitmodel = ViT(
    seq_len=256,
    patch_size=16,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
          )


if torch.cuda.is_available():
    vitmodel = vitmodel.cuda()

loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# learning_rate = 1e-2 相当于(10)^(-2)
learning_rate = 0.01
optim = torch.optim.SGD(vitmodel.parameters(), lr=learning_rate)

total_train_step = 0 #记录训练的次数
total_test_step = 0 # 记录测试的次数
epoch = 50 # 训练的轮数

writer = SummaryWriter("../logs_train")

start_time = time.time()

os.makedirs("model_save/Mobile_Mydatasets", exist_ok=True)

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    # 训练步骤
    for data in train_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = vitmodel(imgs)
        loss = loss_fn(outputs,targets)

        # 优化器优化模型
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤（以测试数据上的正确率来评估模型）
    total_test_loss = 0.0
    # 整体正确个数
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = vitmodel(imgs)
            # 损失
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy,total_test_step)
    total_test_step += 1

    # 保存每一轮训练模型
    torch.save(vitmodel, os.path.join("model_save", "Mobile_Mydatasets", f"mobile_{i}.pth"))
    print("模型已保存")

writer.close()