import os
from PIL import Image
from torchvision.datasets import CIFAR10

# 定义CIFAR-10的类别名称
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 定义保存路径
train_root = 'data/cifar10-imagefolder/train'  # 训练集保存路径
test_root = 'data/cifar10-imagefolder/test'    # 测试集保存路径

# 下载并加载CIFAR-10数据集（不进行转换，保留PIL.Image格式）
train_set = CIFAR10(root='./data/cifar10/', train=True, download=True, transform=None)
test_set = CIFAR10(root='./data/cifar10/', train=False, download=True, transform=None)

# 保存训练集图像到对应目录
for idx, (image, label) in enumerate(train_set):
    class_name = classes[label]
    save_dir = os.path.join(train_root, class_name)
    os.makedirs(save_dir, exist_ok=True)  # 创建类别子目录（如果不存在）
    image.save(os.path.join(save_dir, f'{idx:05d}.png'))  # 保存图像，文件名用5位数字填充

# 保存测试集图像到对应目录
for idx, (image, label) in enumerate(test_set):
    class_name = classes[label]
    save_dir = os.path.join(test_root, class_name)
    os.makedirs(save_dir, exist_ok=True)
    image.save(os.path.join(save_dir, f'{idx:05d}.png'))

# # 示例：使用ImageFolder加载转换后的数据（需定义合适的数据增强和标准化）
# from torchvision import transforms
# from torchvision.datasets import ImageFolder

# # 定义数据预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # 加载训练集和测试集
# train_dataset = ImageFolder(root=train_root, transform=transform)
# test_dataset = ImageFolder(root=test_root, transform=transform)

# # 创建DataLoader
# from torch.utils.data import DataLoader

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)