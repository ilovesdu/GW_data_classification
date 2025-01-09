import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from model import CNNLSTM  # 导入您的模型
from data_preparation import prepare_data  # 导入您的数据准备函数

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据准备
positive_samples_dir = "./positive_cleaned"  # 替换为您的正样本文件夹路径
negative_samples_dir = "./negative_cleaned"  # 替换为您的负样本文件夹路径
X_train, X_test, y_train, y_test = prepare_data(positive_samples_dir, negative_samples_dir,random_state=20)
print(X_test[0])
print(y_test)

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

# 对测试集进行处理
X_test = preprocess_samples(X_test)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # 形状为 (batch, 1)

# 创建数据集和数据加载器
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 加载模型并恢复训练好的权重
model = CNNLSTM(hidden_size=128).to(device)  # 使用与训练时相同的 hidden_size

# 加载最优模型参数
model.load_state_dict(torch.load("./best_model.pth"))
model.eval()  # 设置为评估模式

# 验证模型性能
y_true = []
y_pred = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)  # 输出未经过 Sigmoid
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()  # 转换为概率
        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(probs)

# 计算性能指标
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
acc = accuracy_score(y_true, y_pred_binary)
try:
    auc = roc_auc_score(y_true, y_pred)
except ValueError:
    auc = 0.0  # 如果只有一个类，AUC 无法计算
precision = precision_score(y_true, y_pred_binary, zero_division=0)
recall = recall_score(y_true, y_pred_binary, zero_division=0)

# 打印结果
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
