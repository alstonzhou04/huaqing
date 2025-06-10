# 搭建AlexNet神经网络
import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # 特征提取部分 (参考标准AlexNet架构，适配CIFAR-10的32x32输入)
        self.features = nn.Sequential(
            # 第一层卷积: 输入3通道，输出96通道，卷积核5x5，步长1，填充2
            # (对应原图中的第一层，但调整卷积核和步长适配32x32输入)
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # 第二层卷积: 输入96通道，输出256通道，卷积核5x5，步长1，填充2
            # (对应原图中的第二层)
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            # 第三层卷积: 输入256通道，输出384通道，卷积核3x3，步长1，填充1
            # (对应原图中的第三层)
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 第四层卷积: 输入384通道，输出384通道，卷积核3x3，步长1，填充1
            # (对应原图中的第四层)
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 第五层卷积: 输入384通道，输出256通道，卷积核3x3，步长1，填充1
            # (对应原图中的第五层)
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )
        
        # 自适应平均池化，确保输出大小为4x4 (适配CIFAR-10)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 分类器部分 (参考标准AlexNet架构: 2048->2048->1000，但适配CIFAR-10)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 2048),  # 第一个全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),  # 第二个全连接层
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),  # 输出层 (CIFAR-10为10类)
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 自适应池化
        x = self.avgpool(x)
        # 展平
        x = torch.flatten(x, 1)
        # 分类
        x = self.classifier(x)
        return x


# 测试模型
if __name__ == '__main__':
    # 创建AlexNet模型实例
    alex = AlexNet(num_classes=10)
    
    # 创建输入张量 (batch_size=64, channels=3, height=32, width=32)
    # CIFAR-10图像大小为32x32
    input_tensor = torch.ones((64, 3, 32, 32))
    
    # 前向传播
    output = alex(input_tensor)
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数总数: {sum(p.numel() for p in alex.parameters())}")
    
    # 打印网络结构
    print("\nAlexNet网络结构:")
    print(alex)
