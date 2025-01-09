import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from model import CNNLSTM  # 假设你的模型文件名为 model.py
from model import SimpleCNN
from data_preparation import prepare_data  # 假设你的数据准备文件名为 data_preparation.py
import sys

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据准备
positive_samples_dir = "./data/pos_combined"  # 替换为您的正样本文件夹路径
negative_samples_dir = "./data/negative_combined"  # 替换为您的负样本文件夹路径
X_train, X_test, y_train, y_test = prepare_data(positive_samples_dir, negative_samples_dir)
print(X_test[0])
print(y_test)

# 检查标签分布
num_positive = sum(y_train)
num_negative = len(y_train) - num_positive
print(f"Training labels distribution: {num_positive} positives out of {len(y_train)} samples")

num_positive_test = sum(y_test)
num_negative_test = len(y_test) - num_positive_test
print(f"Testing labels distribution: {num_positive_test} positives out of {len(y_test)} samples")

# 将 (H1, L1) 转换为 (batch, time, channels) 的格式，并进行标准化
def preprocess_samples(samples):
    """
    将 (H1, L1) 样本转换为 (batch, time, channels) 的格式，并进行标准化。
    """
    processed = []
    for combined_data in samples:
        # combined_data 形状应为 (8192, 2)，其中 2 表示 H1 和 L1
        assert combined_data.shape[1] == 2, f"Expected shape (8192, 2), got {combined_data.shape}"
        
        # 获取 H1 和 L1 数据
        h1 = combined_data[:, 0]  # H1 数据
        l1 = combined_data[:, 1]  # L1 数据
        
        # 标准化每个通道
        h1 = (h1 - np.mean(h1)) / (np.std(h1) + 1e-8)
        l1 = (l1 - np.mean(l1)) / (np.std(l1) + 1e-8)
        
        # 堆叠 H1 和 L1，在最后一个维度上形成 (time, channels)
        combined = np.stack((h1, l1), axis=-1)  # 形状应为 (8192, 2)
        processed.append(combined)
    
    # 将列表转换为 NumPy 数组，形状为 (batch, time, channels)
    processed = np.array(processed)
    # 转换为 PyTorch 张量
    return torch.tensor(processed, dtype=torch.float32)

X_train = preprocess_samples(X_train)
X_test = preprocess_samples(X_test)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # 形状为 (batch, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)    # 形状为 (batch, 1)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# 计算 pos_weight
num_positive = torch.sum(y_train)
num_negative = len(y_train) - num_positive
pos_weight = torch.tensor([num_negative / (num_positive + 1e-8)]).to(device)
print(f"pos_weight: {pos_weight.item():.4f}")

# 模型初始化
hidden_size = 128
model = CNNLSTM(hidden_size).to(device)

#初始化模型最后一层的偏置，使初始预测概率稍高于正类
for m in model.classifier.modules():
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0.1)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 使用 pos_weight
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 增大学习率

# 训练函数
def train_model(model, train_loader, test_loader, num_epochs=200):
    best_auc = 0.0
    best_model_path = "./best_model.pth"

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)  # 输出未经过 Sigmoid
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证模型
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)  # 输出未经过 Sigmoid
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()  # 转换为概率
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(probs)

        # 计算指标
        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
        acc = accuracy_score(y_true, y_pred_binary)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = 0.0  # 如果只有一个类，AUC 无法计算
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, ACC: {acc:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # 打印输出概率的统计信息
        print(f"Output probabilities - min: {min(y_pred):.4f}, max: {max(y_pred):.4f}, mean: {np.mean(y_pred):.4f}")

        # 保存最佳模型
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with AUC: {auc:.4f}")

# 开始训练
train_model(model, train_loader, test_loader, num_epochs=100)
