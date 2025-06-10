# 使用AlexNet进行CIFAR-10训练
import time

import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from alex import AlexNet

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="dataset_chen",
                                         train=True,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

test_data = torchvision.datasets.CIFAR10(root="dataset_chen",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True )

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度: {train_data_size}")
print(f"测试数据集的长度: {test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 创建AlexNet网络模型
alexnet = AlexNet(num_classes=10)
alexnet = alexnet.to(device)  # 将模型移动到GPU（如果可用）

print(f"AlexNet模型参数总数: {sum(p.numel() for p in alexnet.parameters()):,}")

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器 - 对于AlexNet，通常使用较小的学习率
learning_rate = 0.001  # 相比原始0.01，使用更小的学习率
optim = torch.optim.SGD(alexnet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# 设置训练网络的一些参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0   # 记录测试的次数
epoch = 20            # AlexNet通常需要更多轮训练

# 添加tensorboard
writer = SummaryWriter("conv_logs/alexnet_train")

# 添加开始时间
start_time = time.time()

print("开始训练AlexNet...")
print(f"总共训练 {epoch} 轮")

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    
    # 训练模式
    alexnet.train()
    epoch_train_loss = 0.0
    
    # 训练步骤
    for data in train_loader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)  # 移动数据到GPU
        
        outputs = alexnet(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optim.zero_grad()  # 梯度清零
        loss.backward()    # 反向传播
        optim.step()

        total_train_step += 1
        epoch_train_loss += loss.item()
        
        if total_train_step % 200 == 0:  # 更频繁地打印训练信息
            print(f"第{total_train_step}步训练的loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 计算每轮的平均训练损失
    avg_train_loss = epoch_train_loss / len(train_loader)
    
    # 计算训练时间
    current_time = time.time()
    print(f"第{i+1}轮训练时间: {current_time - start_time:.2f}秒")
    print(f"第{i+1}轮平均训练损失: {avg_train_loss:.4f}")

    # 评估模式
    alexnet.eval()
    
    # 测试步骤（以测试数据上的正确率来评估模型）
    total_test_loss = 0.0
    total_accuracy = 0  # 整体正确个数
    
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            
            outputs = alexnet(imgs)
            # 损失
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    # 计算测试指标
    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_accuracy / test_data_size
    
    print(f"第{i+1}轮测试集平均loss: {avg_test_loss:.4f}")
    print(f"第{i+1}轮测试集正确率: {test_accuracy:.4f} ({total_accuracy}/{test_data_size})")
    
    # 记录到tensorboard
    writer.add_scalar("test_loss", avg_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    writer.add_scalar("train_loss_epoch", avg_train_loss, total_test_step)
    total_test_step += 1

    # 保存每一轮训练模型
    torch.save(alexnet, f"model_save/alexnet_{i}.pth")
    print(f"第{i+1}轮模型已保存")
    print("-" * 50)

# 计算总训练时间
total_time = time.time() - start_time
print(f"训练完成！总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")

writer.close()
print("TensorBoard日志已保存到 conv_logs/alexnet_train") 