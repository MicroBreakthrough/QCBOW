#此版本是尝试使用流式输入大数据集的版本

import os
import random
import mmap
import chardet
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.utils.data import IterableDataset, DataLoader

from collections import defaultdict
from joblib import Parallel, delayed
import torch.nn.functional as F



os.environ['PYTORCH_CUDA_ALLOW_CONV_ETERMINISTIC_ALGORITHMS'] = '1'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 配置参数
CONFIG = {
    #模型参数
    "quantum_dim": 8,        # 4量子位对应的密度矩阵维度
    "classical_dim":36,     # 经典嵌入维度
    "window_size": 5,

    #学习参数
    "lr": 0.01,            # 学习率
    "epochs": 5,           # 训练轮数
    "grad_clip": 1,      # 梯度裁剪阈值
    "epsilon": 1e-10,        # 数值稳定性系数



    #宏
    "show_progress": True,  #训练时展示进度
    "output_mode":"all",#显式模式 off/basic/all
    "oov_strategy": "skip", # OOV处理策略：skip/zero/random
    "progress_log_interval": 0.1,  # 进度日志间隔 (0.1=10%)
    "log_time_threshold": 0.1,  # 最小日志时间间隔

    #保存
    "save_dir": "saved_models",  # 模型保存目录
    "save_freq": 1,  # 每n个epoch保存一次
    "save_best": True,  # 保存最佳模型
    "max_checkpoints": 5,  # 最大保存检查点数

    #其他
    "quantum_prefix": "quantum",
    "classical_prefix": "classical",

}

class QuantumTextDataset(Dataset):
    def __init__(self, file_path, window_size):
        self.data = []
        self.window_size = window_size
        self._build_dataset(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return context, target

    def _build_dataset(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            buffer = ""
            sentences = []

            while True:
                chunk = mm.read(16 * 1024)  # 16KBchunk
                if not chunk:
                    break
                buffer += chunk.decode('utf-8', errors='ignore')
                lines = buffer.split('\n')
                buffer = lines.pop() if lines else ''
                sentences.extend([line.strip().split() for line in lines])
            mm.close()

        #
        self.vocab = list({word for sentence in sentences for word in sentence})
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        #
        for sentence in sentences:
            for i in range(len(sentence)):
                context = [
                    self.word2idx[sentence[j]]
                    for j in range(max(0, i - self.window_size),
                                   min(len(sentence), i + self.window_size + 1))
                    if j != i
                ]
                if context:
                    self.data.append((context, self.word2idx[sentence[i]]))

class DensityMatrixEmbedding(nn.Module):
    """量子嵌入层，生成合法的密度矩阵"""

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.tril_size = dim * (dim + 1) // 2
        self.embeddings = nn.Embedding(vocab_size, self.tril_size)
        self._init_weights()



    def _init_weights(self):
        nn.init.orthogonal_(self.embeddings.weight)
        with torch.no_grad():
            self.embeddings.weight.mul_(0.01)
            # 对角线初始化
        diag_indices = [i*(i+1)//2 + i for i in range(self.dim)]
        with torch.no_grad():
            diag_vals = torch.linspace(0.1, 0.2, self.dim)
            diag_vals += torch.randn(self.dim) * 0.01  # 添加高斯噪声
            self.embeddings.weight[:, diag_indices] = diag_vals

    def forward(self, indices):
        device = indices.device
        batch_size, seq_len = indices.size()

        tril_params = self.embeddings(indices.view(-1))


        # 构建下三角矩阵
        L = torch.zeros(tril_params.size(0), self.dim, self.dim,
                        device=device)
        tril_idx = torch.tril_indices(self.dim, self.dim, 0)
        L[:, tril_idx[0], tril_idx[1]] = tril_params

        # 正定
        diag = L.diagonal(dim1=1, dim2=2)
        diag = torch.clamp(diag, min=1e-4)  # 原为 CONFIG['epsilon'] (1e-6)
        L = L.clone()
        L.diagonal(dim1=1, dim2=2).copy_(diag)

        return L.view(batch_size, seq_len, self.dim, self.dim)

def compute_density(L):
    """确保半正定性和数值稳定性"""
    # 维度处理
    if L.dim() == 4:  # [batch, seq, dim, dim]
        batch, seq, dim, _ = L.size()
        L = L.view(-1, dim, dim)
    elif L.dim() == 3:  # [batch, dim, dim]
        batch, dim, _ = L.size()
        seq = 1
    else:
        raise ValueError(f"输入张量维度错误: 应为3D或4D, 实际为{L.dim()}D")
    I = CONFIG['epsilon'] * torch.eye(dim, device=L.device).unsqueeze(0)
    rho = torch.bmm(L, L.transpose(1, 2)) + I  # [batch*seq, dim, dim]

    # 特征值修正
    eigenvalues = torch.linalg.eigvalsh(rho)
    min_eigenvalues = eigenvalues.min(dim=-1).values
    correction = torch.clamp(-min_eigenvalues+ 1e-6, min=0)
    rho = rho + correction.view(-1, 1, 1) * torch.eye(dim, device=L.device).unsqueeze(0)

    # 归一化处理
    trace = rho.diagonal(dim1=1, dim2=2).sum(1).view(-1, 1, 1)
    rho = rho / (trace + CONFIG['epsilon'])
    rho = 0.5 * (rho + rho.transpose(-1, -2))

    if seq > 1:
        return rho.view(batch, seq, dim, dim)
    return rho.squeeze(1)

def matrix_sqrt(x):
    U, S, Vh = torch.linalg.svd(x)
    S_reg = S + CONFIG['epsilon']  # 正则化奇异值
    sqrt_S = torch.sqrt(S_reg)
    return (U @ torch.diag_embed(sqrt_S)) @ Vh


def quantum_fidelity(rho, sigma):
    with torch.amp.autocast(device_type='cuda', enabled=False):
        if rho.dim() == 2:
            rho = rho.unsqueeze(0)
        if sigma.dim() == 2:
            sigma = sigma.unsqueeze(0)

        assert rho.dim() == 3 and sigma.dim() == 3, f"输入维度错误: rho={rho.shape}, sigma={sigma.shape}"

        rho = 0.5 * (rho + rho.transpose(-1, -2))  # 确保Hermitian
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))

        sqrt_rho = matrix_sqrt(rho.float())
        product = torch.bmm(sqrt_rho, torch.bmm(sigma.float(), sqrt_rho))
        U, S, Vh = torch.linalg.svd(product)
        S = torch.clamp(S, min=CONFIG['epsilon'],max=1.0-CONFIG['epsilon'])
        fidelity = (torch.sum(torch.sqrt(S), dim=-1)) ** 2
        if torch.any(torch.isnan(S)):
            print(f"检测到NaN奇异值: min={S.min().item()}, max={S.max().item()}")
            S = torch.clamp(S, min=1e-6, max=1.0)
        return fidelity.clamp(min=0.0, max=1.0)

class QuantumCBOW(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.emb = DensityMatrixEmbedding(vocab_size, dim)

    def __repr__(self):
        return f"QuantumCBOW(vocab={self.vocab_size}, dim={self.dim})"  # 调试信息

    def forward(self, contexts, targets):

        contexts = contexts.to(device)
        targets = targets.to(device)
        L_context = self.emb(contexts)
        rho_context = compute_density(L_context)

        # 平均池化
        mask = (contexts != 0).float().view(-1, contexts.size(1), 1, 1)  # [B, S, 1, 1]
        context_rho = (rho_context * mask).sum(dim=1) / (mask.sum(dim=1) + CONFIG['epsilon'])  # [B, d, d]

        L_target = self.emb(targets.unsqueeze(1))  # 添加序列维度 [B, 1] -> [B, 1, d, d]
        rho_target = compute_density(L_target).squeeze(1)  # 移除序列维度 [B, d, d]

        with torch.amp.autocast(device_type='cuda', enabled=False):  # 临时禁用混合精度
            f = quantum_fidelity(context_rho.float(), rho_target.float())


        alpha = 0.1
        loss = -torch.log(f.clamp(min=1e-8)) + alpha * (1 - f)
        return loss.mean()

class ClassicalCBOW(nn.Module):
    """经典CBOW模型"""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size  # 新增属性
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # self.projection = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU()
        # )
        # self.fc = nn.Linear(embed_dim, vocab_size)
        self._init_weights()
        self.to(device)  # 确保模型在设备上

    def _init_weights(self):
        nn.init.xavier_normal_(self.embeddings.weight)
        #nn.init.zeros_(self.fc.bias)

    def forward(self, contexts, targets):
        embeds = self.embeddings(contexts)
        mask = (contexts != 0).float().unsqueeze(-1)
        pooled = (embeds * mask).sum(1) / (mask.sum(1) + 1e-6)
        #projected = self.projection(pooled)
        #return self.fc(projected)
        target_embeds = self.embeddings(targets)  # [batch, embed_dim]
        loss = -nn.functional.cosine_similarity(pooled, target_embeds).mean()
        return loss



