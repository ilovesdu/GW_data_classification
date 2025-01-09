import os
import numpy as np
from sklearn.model_selection import train_test_split


def load_data_from_folder(folder, label):
    """
    从文件夹中加载数据并为所有加载的样本分配给定标签。

    参数:
        folder (str): 包含样本的文件夹（每个样本在其自己的子文件夹中）。
        label (int): 要分配的标签（1 为正类，0 为负类）。

    返回:
        list: 包含 (H1_strain, L1_strain, label) 的列表。
    """
    samples = []
    h1_count = 0
    l1_count = 0

    for sample_dir in sorted(os.listdir(folder)):
        sample_path = os.path.join(folder, sample_dir)
        if not os.path.isdir(sample_path):
            continue

        h1_file = None
        l1_file = None

        for file in os.listdir(sample_path):
            if file.endswith("H1_bandpassed_data.txt"):
                h1_file = os.path.join(sample_path, file)
                h1_count += 1
            elif file.endswith("L1_bandpassed_data.txt"):
                l1_file = os.path.join(sample_path, file)
                l1_count += 1

        if h1_file and l1_file:
            try:
                # 加载数据并提取应变数据（假设第一行为提示信息，删除第一行和第一列）
                h1_data = np.loadtxt(h1_file, skiprows=1)[:, 1:]  # 删除第一行（时间）和第一列（时间信息）
                l1_data = np.loadtxt(l1_file, skiprows=1)[:, 1:]  # 删除第一行（时间）和第一列（时间信息）
                
                # 确保数据是二维的（形状为 (8192, )）
                assert h1_data.ndim == 2 and l1_data.ndim == 2, f"Expected 2D data, got h1_data: {h1_data.ndim}, l1_data: {l1_data.ndim}"
                
                # 堆叠 H1 和 L1 数据，形状为 (8192, 2)
                combined_data = np.stack((h1_data.flatten(), l1_data.flatten()), axis=-1)  # Flatten后堆叠
                #print(f"Stacked data shape (H1, L1): {combined_data.shape}")  # 打印堆叠后的数据维度
                
                samples.append((combined_data, label))
            except Exception as e:
                print(f"Error loading {h1_file} or {l1_file}: {e}")

    print(f"Folder: {folder} -> H1 files: {h1_count}, L1 files: {l1_count}, Matched samples: {len(samples)}")
    return samples

def prepare_data(positive_folder, negative_folder, test_size=0.1, random_state=20, pos_neg_ratio=1):
    """
    准备训练和测试的数据。

    参数:
        positive_folder (str): 包含正样本的文件夹路径。
        negative_folder (str): 包含负样本的文件夹路径。
        test_size (float): 用作测试数据的数据比例。
        random_state (int): 随机种子以保证结果可复现。
        pos_neg_ratio (float): 正负样本的期望比例。

    返回:
        tuple: 训练和测试数据 (X_train, X_test, y_train, y_test)。
    """
    # 加载正负样本
    positive_samples = load_data_from_folder(positive_folder, label=1)
    negative_samples = load_data_from_folder(negative_folder, label=0)

    # 确保正负样本比例
    max_negatives = int(len(positive_samples) / pos_neg_ratio)
    negative_samples = negative_samples[:max_negatives]

    print(f"Positive samples loaded: {len(positive_samples)}")
    print(f"Negative samples loaded (after ratio adjustment): {len(negative_samples)}")

    # 合并并打乱数据
    data = positive_samples + negative_samples
    np.random.shuffle(data)

    # 分割特征和标签
    X = [combined for combined, _ in data]
    y = [label for _, label in data]

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"Positive samples in training: {sum(y_train)}, Negative samples in training: {len(y_train) - sum(y_train)}")
    print(f"Positive samples in testing: {sum(y_test)}, Negative samples in testing: {len(y_test) - sum(y_test)}")

    return X_train, X_test, y_train, y_test

# 示例用法
if __name__ == "__main__":
    positive_samples_dir = "./positive_cleaned"  # 替换为您的正样本文件夹路径
    negative_samples_dir = "./negative_cleaned"  # 替换为您的负样本文件夹路径

    X_train, X_test, y_train, y_test = prepare_data(positive_samples_dir, negative_samples_dir)
