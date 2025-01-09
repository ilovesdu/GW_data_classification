import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from diffusion_model import DiffusionModel
import os
import sys
import numpy as np 
def standardize(data):
    mean = data.mean(dim = 0, keepdim = True)
    standard = data.standard(dim = 0, keepdim = True)
    return (data - mean) / standard

def linear_noise_schedule(T, beta_start=1e-4, beta_end=2e-2):
    """
    生成线性噪声调度的beta序列。
    参数:
        T (int): 总时间步数。
        beta_start (float): 初始beta值。
        beta_end (float): 终止beta值。
        
    返回:
        betas (torch.Tensor): 线性递增的beta序列。
    """
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

def train_diffusion_model(data_path, output_dir, epochs=50, lr=1e-4, batch_size=16, device="cuda", T=1000):
    """
    训练扩散模型以生成正样本。

    参数:
        data_path (str): .pt 文件路径，包含正样本数据。
        output_dir (str): 保存模型的目录路径。
        epochs (int): 训练的轮数。
        lr (float): 学习率。
        batch_size (int): 每批样本数量。
        device (str): 使用的设备（默认为 CUDA）。
        T (int): 总时间步数。
    """
    # 加载数据
    train_data = torch.load(data_path)["X"]  # 加载字典中的 "X" 数据
    train_data = standardize(train_data)
    dataloader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = DiffusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs(output_dir, exist_ok=True)

    # 定义线性噪声调度
    betas, alphas, alphas_cumprod = linear_noise_schedule(T)
    betas, alphas, alphas_cumprod = betas.to(device), alphas.to(device), alphas_cumprod.to(device)

    # 预计算sqrt(alphas_cumprod)和sqrt(1 - alphas_cumprod)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).view(-1, 1, 1)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).view(-1, 1, 1)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device).permute(0, 2, 1)  # (B, 8192, 2) -> (B, 2, 8192)
            batch_size_current = x.size(0)
            
            # 随机选择时间步 t
            t = torch.randint(0, T, (batch_size_current,), device=device).long()
            
            # 获取对应的sqrt(alphas_cumprod)和sqrt(1 - alphas_cumprod)
            sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[:, :, :].repeat(batch_size_current, 1, 1)[range(batch_size_current), :, :] * sqrt_alphas_cumprod[t].view(-1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[:, :, :].repeat(batch_size_current, 1, 1)[range(batch_size_current), :, :] * sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
            
            # 生成噪声
            noise = torch.randn_like(x)
            
            # 生成噪声添加后的样本
            noisy_x = sqrt_alphas_cumprod[t].view(-1, 1, 1) * x + sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise
            
            optimizer.zero_grad()

            # 预测噪声并计算损失
            predicted_noise = model(noisy_x, t)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), os.path.join(output_dir, "diffusion_model.pth"))

if __name__ == "__main__":
    data_file = "./pos_combined.pt"  # 替换为您的 .pt 文件路径
    output_directory = "./diffusion_output"  # 保存训练模型的路径

    train_diffusion_model(data_file, output_directory, epochs=30)
