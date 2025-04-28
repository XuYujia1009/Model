import os
import torch
import time
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from calc_miou import compute_mIoU, score
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from channel_unet import self_net

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_training_curves(train_loss, val_loss, miou_list, save_path):
    epochs = range(1, len(train_loss) + 1)

    # 创建图像
    plt.figure(figsize=(12, 6))

    # 训练损失和验证损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r-', label='Training Loss')
    plt.plot(epochs, val_loss, 'b-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # mIoU 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, miou_list, 'g-', label='mIoU')
    plt.title('mIoU over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()

    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path)


# 定义数据集
class SegmentationDataset(Dataset):
    def __init__(self, images_list, labels_list, transform=None):
        self.images_list = images_list
        self.labels_list = labels_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        mask_path = self.labels_list[idx]

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, torch.tensor(mask, dtype=torch.long)


# 训练模型
def train_model(model, dataloader, criterion, optimizer, device, num_epochs, val_dataloader):
    model.train()
    best_miou = 0
    best_score = 0
    train_loss=[]
    val_loss=[]
    miou_list=[]
    gt_dir='result/test_ground_truths'
    pred_dir='result/test_predictions'
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        output = f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}'
        print(output)
        _,avg_loss=validate_model(model, val_dataloader, device,criterion)
        train_loss.append(epoch_loss)
        val_loss.append(avg_loss)
        save_predictions_as_npy(model, val_dataloader, pred_dir, device)
        save_ground_truths_as_npy(val_dataloader, gt_dir)
        pre_IoU = compute_mIoU(gt_dir, pred_dir)
        mean_iou = np.nanmean(pre_IoU)
        miou_list.append(mean_iou)
        if mean_iou > best_miou:
            best_miou = mean_iou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'model.pth')  # 保存最佳模型的权重
    
    plot_training_curves(train_loss, val_loss, miou_list, f'result/my.png')
    print(f'训练结束，最佳模型在第 {best_epoch} 轮，mIoU: {best_miou:.4f}，best_score：{best_score}')
    print('训练结束')

# 验证模型
def validate_model(model, dataloader, device, criterion):
    model.eval()
    total_time = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            start_time = time.time()  # 计时
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            total_time += time.time() - start_time  # 计算时间
    avg_time_per_image = total_time / len(dataloader.dataset)
    fps = 1 / avg_time_per_image
    avg_loss = total_loss / len(dataloader)
    print(f"test_loss:{avg_loss}")

    return fps,avg_loss

# 保存分割结果
def save_predictions_as_npy(model, dataloader, output_dir, device):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            for j in range(predictions.shape[0]):
                img_filename = os.path.basename(dataloader.dataset.images_list[i * dataloader.batch_size + j]) 
                output_path = os.path.join(output_dir, f"prediction_{img_filename.replace('.jpg', '.npy')}")
                np.save(output_path, predictions[j])

# 保存Ground Truth
def save_ground_truths_as_npy(dataloader, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, (_, labels) in enumerate(dataloader):
        for j in range(labels.shape[0]):
            img_filename = os.path.basename(dataloader.dataset.images_list[i * dataloader.batch_size + j])
            output_path = os.path.join(output_dir, f"ground_truth_{img_filename.replace('.jpg', '.npy')}")
            np.save(output_path, labels[j].cpu().numpy())

# 计算模型参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_results_to_txt(results, filename):
    with open(filename, 'w') as f:
        f.write(str(results))


def dice_loss_multiclass(pred, target, smooth=1e-6):
    # 对预测结果应用 softmax 函数，得到每个类别的概率
    pred = F.softmax(pred, dim=1)
    
    # 获取类别数量（假设为 4 类）
    num_classes = pred.size(1)
    
    # 初始化总损失
    loss = 0.0
    
    for c in range(num_classes):
        # 为当前类别创建一个二进制的目标掩码
        target_c = (target == c).float()
        
        # 计算当前类别的 Dice 系数
        intersection = (pred[:, c] * target_c).sum()
        union = pred[:, c].sum() + target_c.sum()
        
        # 计算当前类别的 Dice 损失
        dice_c = (2. * intersection + smooth) / (union + smooth)
        
        # 将当前类别的 Dice 损失加到总损失中
        loss += 1 - dice_c
    
    # 返回所有类别的平均 Dice 损失
    return loss / num_classes

def iou_loss_multiclass(pred, target, smooth=1e-6):
    # 对预测结果应用 softmax 函数，得到每个类别的概率
    pred = F.softmax(pred, dim=1)
    
    # 获取类别数量（假设为 4 类）
    num_classes = pred.size(1)
    
    # 初始化总损失
    loss = 0.0
    for c in range(num_classes):
        # 创建二进制的目标掩码，当前类别的掩码
        target_c = (target == c).float()
        
        # 计算交集：预测为当前类别且目标为当前类别的像素点数量
        intersection = (pred[:, c] * target_c).sum()
        
        # 计算并集：预测为当前类别的像素点数量 + 目标为当前类别的像素点数量 - 交集部分
        union = pred[:, c].sum() + target_c.sum() - intersection
        
        # 计算当前类别的 IoU
        iou_c = (intersection + smooth) / (union + smooth)
        
        # 将当前类别的 IoU 损失加到总损失中
        loss += 1 - iou_c
    
    # 返回所有类别的平均 IoU 损失
    return loss / num_classes

def hybrid_loss(pred, target, alpha=0.3, smooth=1e-6):
    
    ce_loss=nn.CrossEntropyLoss()
    focal_loss=focal_loss_multiclass
    iou_loss=iou_loss_multiclass
    dice_loss=dice_loss_multiclass
    # 混合损失 = alpha * 交叉熵损失 + (1 - alpha) * IoU损失
    #total_loss = alpha * ce_loss(pred, target) + (1 - alpha-0.5) * focal_loss(pred, target)+0.2*dice_loss(pred, target)+0.2*iou_loss(pred, target)
    total_loss = alpha * ce_loss(pred, target) + (1 - alpha-0.3) * focal_loss(pred, target)+0.3*dice_loss(pred, target)
    return total_loss

def focal_loss_multiclass(pred, target, alpha=0.25, gamma=2.0, smooth=1e-6):
    # 确保 target 和 pred 在同一设备上
    device = pred.device
    target = target.to(device)
    
    # 对预测结果应用 softmax 函数，得到每个类别的概率
    pred = F.softmax(pred, dim=1)
    
    # 获取类别数量
    num_classes = pred.size(1)
    
    # 创建一个与 target 相同形状的 one-hot 编码的目标
    # 将 target 从 (batch_size, height, width) 转换为 (batch_size, num_classes, height, width)
    target_one_hot = torch.eye(num_classes, device=device)[target].permute(0, 3, 1, 2)

    # 计算交叉熵的预测概率
    p_t = (pred * target_one_hot).sum(dim=1) + smooth  # 对每个类别计算 p_t（目标类别的预测概率）

    # 计算 Focal Loss
    loss = -alpha * (1 - p_t) ** gamma * torch.log(p_t)
    
    # 平均损失
    return loss.mean()

# 主程序
if __name__ == '__main__':
    # 设置随机种子
    def set_seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    # 文件夹路径
    data_dir = 'datasets'
    img_dir = os.path.join(data_dir, 'img')
    lab_dir = os.path.join(data_dir, 'lab')

    # 获取所有图像和标签文件的路径
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
    lab_files = sorted([os.path.join(lab_dir, f) for f in os.listdir(lab_dir) if f.endswith('.png')])

    # 数据集划分为80%训练集，20%验证集
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(img_files, lab_files, test_size=0.2, random_state=42)

    # 数据增强和预处理
    train_transform = A.Compose([
        A.Resize(200, 200),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(200, 200),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 数据加载
    train_dataset = SegmentationDataset(train_imgs, train_labels, transform=train_transform)
    val_dataset = SegmentationDataset(val_imgs, val_labels, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 模型设置
    model = self_net().to(device)
    #criterion = nn.CrossEntropyLoss()
    criterion = hybrid_loss
    #criterion = focal_loss_multiclass
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 1

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device, num_epochs, val_loader)

    self_model1=self_net()
    self_model1.load_state_dict(torch.load('model.pth'))
    self_model1.to(device)
    self_fps,_ = validate_model(self_model1, val_loader, device,criterion)
    self_params = count_parameters(self_model1)

    save_predictions_as_npy(self_model1, val_loader, 'result/test_predictions', device)
    pred_dir = 'result/test_predictions'
    gt_dir = 'result/test_ground_truths'
    pre_IoU = compute_mIoU(gt_dir, pred_dir)
    
    # 将结果保存到字典中
    results = {
        "SelfNet": {
            "Class1 IoU": pre_IoU[0],
            "Class2 IoU": pre_IoU[1],
            "Class3 IoU": pre_IoU[2],
            "mIoU": np.nanmean(pre_IoU),
            "FPS": self_fps,
            "Model Parameters": self_params
        }
    }
    
    # 保存结果到txt文件
    save_results_to_txt(results, '关键指标数据文档.txt')
