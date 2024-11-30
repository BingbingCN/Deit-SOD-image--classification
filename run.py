import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.data import create_transform
from tqdm import tqdm
from model import SODModel_DeiT
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# 定义参数
class args:
    shuffle = True
    data_path = 'cifar-100-python'
    res_mod_path = 'pretrained/best-model_epoch-312_mae-0.0448_loss-0.1527.pth'
    batch_size = 1
    num_workers = 0
    nb_classes = 100
    input_size = 224
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据增强
    aa = 'rand-m9-mstd0.5-inc1'
    color_jitter = 0.4
    train_interpolation = 'bicubic'
    reprob = 0.25
    remode = 'pixel'
    recount = 1

    # train
    lr = 5e-4
    epochs = 400



# 构建数据变换
def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform
    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(transforms.Resize(size, interpolation=3))
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


# 计算Top-k准确率
def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())  # 总参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数数量

    return {
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params
    }

if __name__ == '__main__':
    # 数据集加载
    train_transform = build_transform(is_train=True, args=args)
    val_transform = build_transform(is_train=False, args=args)

    train_dataset = datasets.CIFAR100(root=args.data_path, train=True, transform=train_transform, download=True)
    val_dataset = datasets.CIFAR100(root=args.data_path, train=False, transform=val_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 模型初始化
    model = SODModel_DeiT(device=args.device, num_classes=args.nb_classes, res_mod_path=args.res_mod_path).to(args.device)

    param_stats = count_model_params(model)
    for key, value in param_stats.items():
        print(f"{key}: {value:,}")

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 输出路径
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    best_top1_acc = 0.0
    log_file = os.path.join(output_dir, "training_log.txt")

    # 训练和验证
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0.0
        correct_top1, correct_top5, total = 0, 0, 0

        # 训练循环
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [Training]")):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()

            outputs,_ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)

            # 计算准确率
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            correct_top1 += acc1.item() * targets.size(0) / 100.0
            correct_top5 += acc5.item() * targets.size(0) / 100.0


        train_loss /= len(train_loader)
        train_top1_acc = correct_top1 / total * 100
        train_top5_acc = correct_top5 / total * 100

        # 验证循环
        model.eval()
        val_loss = 0.0
        correct_top1, correct_top5, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch} [Validation]"):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                total += targets.size(0)

                # 计算准确率
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                correct_top1 += acc1.item() * targets.size(0) / 100.0
                correct_top5 += acc5.item() * targets.size(0) / 100.0


        val_loss /= len(val_loader)
        val_top1_acc = correct_top1 / total * 100
        val_top5_acc = correct_top5 / total * 100

        # 保存最优模型
        if val_top1_acc > best_top1_acc:
            best_top1_acc = val_top1_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        # 记录日志
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Train Loss: {train_loss:.4f}, Top-1 Accuracy: {train_top1_acc:.2f}%, Top-5 Accuracy: {train_top5_acc:.2f}%\n")
            f.write(f"  Val Loss: {val_loss:.4f}, Top-1 Accuracy: {val_top1_acc:.2f}%, Top-5 Accuracy: {val_top5_acc:.2f}%\n\n")

        print(f"Epoch {epoch} completed.")
        print(f"  Train Loss: {train_loss:.4f}, Top-1 Accuracy: {train_top1_acc:.2f}%, Top-5 Accuracy: {train_top5_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Top-1 Accuracy: {val_top1_acc:.2f}%, Top-5 Accuracy: {val_top5_acc:.2f}%")

