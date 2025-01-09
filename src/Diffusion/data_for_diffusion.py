import os
import torch
import numpy as np


def load_positive_data_from_folder_recursive(folder):
    """
    从嵌套文件夹中递归加载正样本数据。

    参数:
        folder (str): 包含正样本的文件夹路径。

    返回:
        list: 包含 (H1_strain, L1_strain) 的列表。
    """
    samples = []
    h1_count = 0
    l1_count = 0

    for root, dirs, files in os.walk(folder):
        h1_file = None
        l1_file = None

        for file in files:
            if file.endswith("H1_bandpassed_data.txt"):
                h1_file = os.path.join(root, file)
                h1_count += 1
            elif file.endswith("L1_bandpassed_data.txt"):
                l1_file = os.path.join(root, file)
                l1_count += 1

        if h1_file and l1_file:
            try:
                # 加载数据并提取应变数据（假设第一行为提示信息，删除第一行和第一列）
                h1_data = np.loadtxt(h1_file, skiprows=1)[:, 1:]  # 删除第一行（时间）和第一列（时间信息）
                l1_data = np.loadtxt(l1_file, skiprows=1)[:, 1:]  # 删除第一行（时间）和第一列（时间信息）
                
                # 确保数据是二维的
                assert h1_data.ndim == 2 and l1_data.ndim == 2, f"Expected 2D data, got h1_data: {h1_data.ndim}, l1_data: {l1_data.ndim}"
                
                # 堆叠 H1 和 L1 数据，形状为 (8192, 2)
                combined_data = np.stack((h1_data.flatten(), l1_data.flatten()), axis=-1)
                samples.append(combined_data)
            except Exception as e:
                print(f"Error loading {h1_file} or {l1_file}: {e}")

    print(f"Folder: {folder} -> H1 files: {h1_count}, L1 files: {l1_count}, Matched samples: {len(samples)}")
    return samples


def prepare_positive_data_and_save_as_pt(positive_folder, output_file):
    """
    准备正样本数据并保存为 .pt 文件。

    参数:
        positive_folder (str): 包含正样本的文件夹路径。
        output_file (str): 保存 .pt 文件的路径。
    """
    # 加载正样本
    positive_samples = load_positive_data_from_folder_recursive(positive_folder)

    print(f"Positive samples loaded: {len(positive_samples)}")

    # 转换为 PyTorch 张量并保存为 .pt 文件
    torch.save(
        {"X": torch.tensor(np.array(positive_samples), dtype=torch.float32)},
        output_file,
    )
    print(f"Positive samples saved to {output_file}.")


# 示例用法
if __name__ == "__main__":
    positive_samples_dir = "../data/pos_combined"  # 替换为您的正样本文件夹路径
    output_pt_file = "./pos_combined.pt"  # 保存正样本的路径

    prepare_positive_data_and_save_as_pt(positive_samples_dir, output_pt_file)
