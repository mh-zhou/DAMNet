import os
import torch
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
from tqdm import tqdm
from PIL import Image
import time
from model import MobileV2Net  
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.img_label_list = self.read_from_txt(txt_file)
        self.transform = transform

    def read_from_txt(self, txt_file):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        img_label_list = [(line.split()[0], int(line.split()[1])) for line in lines]
        return img_label_list

    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, idx):
        img_path, label = self.img_label_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 加载数据
train_dataset = CustomDataset(txt_file='split_data/train.txt', transform=data_transform)
val_dataset = CustomDataset(txt_file='split_data/val.txt', transform=data_transform)
test_dataset = CustomDataset(txt_file='split_data/test.txt', transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# 验证模型性能
def validate_model(model, loader):
    model.eval()
    acc = 0.0  # 累积准确数/epoch
    with torch.no_grad():
        t1 = time.time()
        for images, labels in tqdm(loader, desc="Validating model"):
            outputs = model(images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.sum(predict_y == labels.to(device)).item()
        accuracy = acc / len(loader.dataset)
        print(f'Validation Accuracy: {accuracy:.3f}, Time: {time.time() - t1:.3f}')
    return accuracy


def count_sparsity(model: torch.nn.Module, p=True):
    sum_zeros_num = 0
    sum_weights_num = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            zeros_elements = (module.weight == 0).sum().item()
            weights_elements = module.weight.numel()
            sum_zeros_num += zeros_elements
            sum_weights_num += weights_elements
            if p:
                print(f"Sparsity in {name}.weights: {100 * zeros_elements / weights_elements:.2f}%")
    print(f"Global sparsity: {100 * sum_zeros_num / sum_weights_num:.2f}%")


def main():
    weights_path = "experiencedata/ours/epoch200/best_model_weights.pth"
    model = MobileV2Net(class_num=4).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1  # 设定训练的 epoch 数

    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
        # 每个epoch结束后在验证集上评估模型
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        validate_model(model, val_loader)
    
    # 裁剪模型
    parameters_to_prune = [(module, 'weight') for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)

    # 统计裁剪后的稀疏性
    count_sparsity(model)

    # 验证裁剪后的模型在验证集上的性能
    print("Validating pruned model on val dataset...")
    validate_model(model, val_loader)

if __name__ == '__main__':
    main()
