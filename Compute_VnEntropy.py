import os
import random
import mmap
import chardet
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.utils.data import IterableDataset, DataLoader

from collections import defaultdict
from itertools import islice



os.environ['PYTORCH_CUDA_ALLOW_CONV_ETERMINISTIC_ALGORITHMS'] = '1'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True)
print(f"Using device: {device}")

# 配置参数
CONFIG = {
    "num_workers": 8,  # 多进程加载
    "prefetch_factor": 2,  # 预取批次
    "max_seq_len": 128,  # 最大序列长度
    "gradient_accumulation_steps": 4,  # 梯度累积步数
    "vocab_limit": 1000000,

    "quantum_dim": 8,        # 3量子位对应的密度矩阵维度
    "classical_dim": 36,     # 经典嵌入维度
    "window_size": 5,
    "batch_size": 32,

    "lr": 0.001,            # 学习率
    "epochs": 5,           # 训练轮数
    "grad_clip": 0.5,       # 梯度裁剪阈值
    "epsilon": 1e-6,        # 数值稳定性系数


    "show_progress":False,  #训练时展示进度
    "oov_strategy": "skip", # OOV处理策略：skip/zero/random

    "save_dir": "saved_models",  # 模型保存目录
    "save_freq": 10,  # 每5个epoch保存一次
    "save_best": False,  # 根据验证损失保存最佳模型
    "quantum_prefix": "quantum",  # 量子模型前缀
    "classical_prefix": "classical",
    "max_checkpoints": 5,           # 新增：最大保存检查点数

    "data_paths": {
        'WordSim-353': '/root/autodl-fs/combined.csv',
        'WordSim-Similarity': '/root/autodl-fs/wordsim_similarity_goldstandard.txt',
        'WordSim-Relatedness': '/root/autodl-fs/wordsim_relatedness_goldstandard.txt',
        'MEN': '/root/autodl-fs/MEN_dataset_natural_form_full.csv',
        'Rubenstein-Goodenough': '/root/autodl-fs/rg65.csv',
        #'TOEFL': 'data/TOEFL.txt'
    }
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class StreamQuantumTextDataset(IterableDataset):
    def __init__(self, file_path, window_size):
        self.file_path = file_path
        self.window_size = window_size
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.file_size = os.path.getsize(file_path)
        self._build_vocab()

    def _build_vocab(self):
        """重构词汇表构建逻辑"""
        word_counts = defaultdict(int)

        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="First pass - counting words"):
                for word in line.strip().split():
                    word_counts[word] += 1


        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        self.word2idx = {'<PAD>': 0}  # 显式保留0给填充符
        idx = 1
        for word, count in sorted_words[:CONFIG['vocab_limit'] - 1]:
            self.word2idx[word] = idx
            idx += 1

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.file_size
        else:
            per_worker = self.file_size // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else self.file_size

        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mm.seek(start)
            buffer = ""
            while mm.tell() < end:
                chunk = mm.read(8192)
                buffer += chunk.decode('utf-8', errors='ignore')
                lines = buffer.split('\n')
                buffer = lines.pop()

                for line in lines:
                    sentence = line.strip().split()
                    for i in range(len(sentence)):
                        word = sentence[i]
                        word_idx = self.word2idx.get(word, 0)
                        context = [
                            self.word2idx.get(sentence[j], 0)
                            for j in range(max(0, i - self.window_size),
                                            min(len(sentence), i + self.window_size + 1))
                            if j != i
                        ][:CONFIG['max_seq_len']]
                        if context:
                            current_pos = mm.tell()
                            yield (context, word_idx, current_pos)

def collate_fn(batch):

    if len(batch[0]) == 3:
        contexts, targets, positions = zip(*batch)
        max_pos = max(positions)
    else:
        contexts, targets = zip(*batch)
        max_pos = None

    lengths = [min(len(c), CONFIG['max_seq_len']) for c in contexts]
    max_len = max(lengths)
    padded = torch.zeros(len(contexts), max_len, dtype=torch.long)
    for i, (c, l) in enumerate(zip(contexts, lengths)):
        padded[i, :l] = torch.tensor(c[:l], dtype=torch.long)

    if max_pos is not None:
        return padded, torch.tensor(targets, dtype=torch.long), max_pos
    else:
        return padded, torch.tensor(targets, dtype=torch.long)

def validate_dataset_file(file_path):
    """数据集文件验证"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件 {file_path} 不存在")
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"数据集文件 {file_path} 为空文件")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"无读取权限：{file_path}")

def detect_delimiter(line):
    """修正拼写错误"""
    delimiters = [',', '\t', ':', ';', '|']
    counts = {delim: line.count(delim) for delim in delimiters}
    return max(counts, key=counts.get, default=None)

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
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip().split() for line in f]


        self.vocab = list({word for sentence in sentences for word in sentence})
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

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
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)


        diag_indices = [i * (i + 1) // 2 + i for i in range(self.dim)]
        with torch.no_grad():
            diag_vals = torch.linspace(0.1, 1.0, self.dim)
            self.embeddings.weight[:, diag_indices] = diag_vals.repeat(
                self.embeddings.weight.size(0), 1)

    def forward(self, indices):
        device = indices.device
        batch_size, seq_len = indices.size()


        tril_params = self.embeddings(indices.view(-1))

        L = torch.zeros(tril_params.size(0), self.dim, self.dim,
                        device=device)
        tril_idx = torch.tril_indices(self.dim, self.dim, 0)
        L[:, tril_idx[0], tril_idx[1]] = tril_params

        diag = L.diagonal(dim1=1, dim2=2)
        diag = torch.clamp(diag, min=1e-4)
        L = L.clone()
        L.diagonal(dim1=1, dim2=2).copy_(diag)

        return L.view(batch_size, seq_len, self.dim, self.dim)


def compute_density(L):
    """确保半正定性和数值稳定性"""

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


    eigenvalues = torch.linalg.eigvalsh(rho)
    min_eigenvalues = eigenvalues.min(dim=-1).values


    correction = torch.clamp(-min_eigenvalues, min=0) + 1e-6
    rho = rho + correction.view(-1, 1, 1) * torch.eye(dim, device=L.device).unsqueeze(0)

    trace = rho.diagonal(dim1=1, dim2=2).sum(1).view(-1, 1, 1)
    rho = rho / (trace + CONFIG['epsilon'])


    if seq > 1:
        return rho.view(batch, seq, dim, dim)
    return rho.squeeze(1)

def matrix_sqrt(x):
    """带正则化的矩阵平方根"""
    U, S, Vh = torch.linalg.svd(x)
    S_reg = S + CONFIG['epsilon']  # 正则化奇异值
    sqrt_S = torch.sqrt(S_reg)
    return (U @ torch.diag_embed(sqrt_S)) @ Vh


def quantum_fidelity(rho, sigma):
    if rho.dim() == 2:
        rho = rho.unsqueeze(0)
    if sigma.dim() == 2:
        sigma = sigma.unsqueeze(0)

    assert rho.dim() == 3 and sigma.dim() == 3, f"输入维度错误: rho={rho.shape}, sigma={sigma.shape}"

    sqrt_rho = matrix_sqrt(rho)
    product = torch.bmm(sqrt_rho, torch.bmm(sigma, sqrt_rho))
    U, S, Vh = torch.linalg.svd(product)
    fidelity = torch.sum(torch.sqrt(S + CONFIG['epsilon']), dim=-1)
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
        # 上下文处理 [B, S] -> [B, S, d, d]
        contexts = contexts.to(device)
        targets = targets.to(device)

        L_context = self.emb(contexts)
        rho_context = compute_density(L_context)

        # 带掩码的平均池化
        mask = (contexts != 0).float().view(-1, contexts.size(1), 1, 1)
        context_rho = (rho_context * mask).sum(dim=1) / mask.sum(dim=1)

        # 目标处理 [B] -> [B, d, d]
        L_target = self.emb(targets.unsqueeze(1))
        rho_target = compute_density(L_target).squeeze(1)

        f = quantum_fidelity(context_rho, rho_target)
        return -torch.log(f.clamp(min=1e-8))

def save_model(model, optimizer, epoch, model_type, is_best=False):



    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    prefix = CONFIG[f"{model_type}_prefix"]

    save_dim = model.dim if hasattr(model, 'dim') else CONFIG['quantum_dim']
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'model_type': model_type,
        'config': CONFIG,
        'vocab_size': model.vocab_size,
        'optimizer_state': optimizer.state_dict(),
        'random_seed': torch.initial_seed(),
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
        'embeddings': {
            'quantum': model.emb.state_dict() if model_type == 'quantum' else None,
            'classical': model.embeddings.state_dict() if model_type == 'classical' else None
        },
        'training_metadata': {
            'loss_history': [],
            'grad_norms': []
        }
    }

    if model_type == 'quantum':
        state['dim'] = model.dim


    filename = f"{prefix}_best.pth" if is_best else f"{prefix}_epoch{epoch}.pth"
    save_path = os.path.join(CONFIG['save_dir'], filename)

    torch.save(state, save_path)


    print(f" {model_type.capitalize()}模型保存成功 -> {save_path}")



def load_model(checkpoint_path, dataset=None):

    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    if checkpoint['model_type'] == 'quantum':
        model = QuantumCBOW(
            vocab_size=checkpoint['vocab_size'],
            dim=checkpoint['dim']
        ).to(device)
        model.emb.load_state_dict(checkpoint['embeddings']['quantum'])
    else:
        model = ClassicalCBOW(
            vocab_size=checkpoint['vocab_size'],
            embed_dim=CONFIG['classical_dim']
        ).to(device)
        model.embeddings.load_state_dict(checkpoint['embeddings']['classical'])


    model.load_state_dict(checkpoint['state_dict'])


    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])


    torch.manual_seed(checkpoint['random_seed'])
    np.random.set_state(checkpoint['numpy_random_state'])
    random.setstate(checkpoint['python_random_state'])

    model.metadata = checkpoint.get('training_metadata', {})

    if checkpoint['model_type'] == 'quantum':
        model.emb.load_state_dict(checkpoint['embeddings']['quantum'])
    else:
        model.embeddings.load_state_dict(checkpoint['embeddings']['classical'])

    cache = EmbeddingCache(model, dataset) if dataset else None
    return model, cache

class EmbeddingCache:
    """嵌入结果缓存系统"""

    def __init__(self, model, dataset):
        self.model = model
        self.word2idx = dataset.word2idx
        self.idx2word = dataset.idx2word
        self.cache = {}

        device = next(model.parameters()).device
        # 预缓存所有词嵌入
        with torch.no_grad():
            indices = torch.arange(len(self.word2idx)).to(device)
            if isinstance(model, QuantumCBOW):
                L = model.emb(indices.unsqueeze(1))
                self.embeddings = compute_density(L).squeeze(1)
            else:
                self.embeddings = model.embeddings(indices)

            for idx, word in self.idx2word.items():
                self.cache[word] = {
                    'index': idx,
                    'embedding': self.embeddings[idx].cpu().numpy(),
                    'stats': {
                        'norm': np.linalg.norm(self.embeddings[idx].cpu().numpy()),
                        'mean': np.mean(self.embeddings[idx].cpu().numpy())
                    }
                }

    def visualize_embedding(self, word, ax=None):
        """嵌入可视化"""
        embed = self.get_embedding(word)
        if embed is None:
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        if isinstance(embed, torch.Tensor):
            data = embed.cpu().numpy()
        else:
            data = embed

        if data.ndim == 2:  # 量子嵌入矩阵可视化
            im = ax.imshow(data, cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'Density Matrix for "{word}"')
        else:  # 经典嵌入降维可视化
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(data.reshape(1, -1))
            ax.scatter(reduced[:, 0], reduced[:, 1], s=100)
            ax.set_title(f'PCA Projection of "{word}"')

        return ax
    def get_embedding(self, word):
        """安全获取嵌入"""
        entry = self.cache.get(word)
        if entry is None:
            return self._handle_oov()
        return torch.tensor(entry['embedding']).to(device)

    def _handle_oov(self):
        """处理OOV策略"""
        if CONFIG['oov_strategy'] == 'zero':
            return torch.zeros_like(self.embeddings[0])
        elif CONFIG['oov_strategy'] == 'random':
            return torch.randn_like(self.embeddings[0])
        return None

    def save_to_text(self, filename):
        """将词向量保存到文本文件"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            for word in self.idx2word.values():
                idx = self.word2idx[word]
                vec = self.embeddings[idx].cpu().numpy()
                if isinstance(self.model, QuantumCBOW):
                    # 将密度矩阵展平存储
                    vec_str = " ".join(map(str, vec.flatten()))
                else:
                    vec_str = " ".join(map(str, vec))
                f.write(f"{word} {vec_str}\n")
        print(f"词向量已保存至 {filename}")

    @classmethod
    def load_from_text(cls, filename, model_type):
        """从文本文件加载词向量"""
        word_vectors = {}
        with open(filename, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="加载词向量"):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                vector = np.array(list(map(float, parts[1:])))
                if model_type == 'quantum':
                    dim = int(np.sqrt(len(vector)))
                    vector = vector.reshape(dim, dim)
                word_vectors[word] = vector
        return word_vectors


def train_model(model, dataloader, optimizer, model_type):
    best_loss = float('inf')
    model.train()
    start_time = time.time()

    is_streaming = hasattr(dataloader.dataset, 'file_size')
    if is_streaming:
        file_size = dataloader.dataset.file_size
        max_processed_pos = 0
    else:
        total_batches = len(dataloader)

    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        processed_batches = 0
        dataloader_tqdm = tqdm(
            dataloader,
            desc=f'Epoch {epoch + 1}/{CONFIG["epochs"]} ({model_type})',
            leave=False,
            disable=not CONFIG['show_progress'],
            dynamic_ncols=True,
            bar_format='{l_bar}{bar:20}| {n_fmt} batches [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )

        for batch in dataloader_tqdm:
            if len(batch) == 3:
                contexts, targets, batch_max_pos = batch
                if batch_max_pos > max_processed_pos:
                    max_processed_pos = batch_max_pos
                progress = max_processed_pos / file_size
            else:
                contexts, targets = batch
                progress = processed_batches / total_batches

            contexts = contexts.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            if model_type == 'quantum':
                loss = model(contexts, targets).mean()
            else:
                outputs = model(contexts)
                loss = nn.functional.cross_entropy(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            optimizer.step()

            total_loss += loss.item()
            processed_batches += 1
            elapsed_time = time.time() - start_time


            postfix = {'batch_loss': f"{loss.item():.4f}"}
            if is_streaming:
                if progress > 0:
                    remaining_time = (elapsed_time / progress) - elapsed_time
                else:
                    remaining_time = float('inf')
                postfix['progress'] = f'{progress:.1%}'
                postfix['remaining'] = f'{remaining_time / 60:.1f}min'
            else:
                postfix['progress'] = f'{processed_batches}/{total_batches}'

            dataloader_tqdm.set_postfix(postfix)

        avg_loss = total_loss / processed_batches
        if CONFIG['save_best'] and avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, optimizer, epoch + 1, model_type, is_best=True)

        dataloader_tqdm.close()


def validate(model, val_loader):
    """验证函数"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for contexts, targets in val_loader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            outputs = model(contexts, targets)
            total_loss += outputs.mean().item()
    return total_loss / len(val_loader)

def evaluate(model, test_data, dataset, is_quantum=True,
           cache=None, word_vectors_file=None):
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    if word_vectors_file:
        model_type = 'quantum' if is_quantum else 'classical'
        word_vectors = EmbeddingCache.load_from_text(word_vectors_file, model_type)
        cache = None  # 覆盖原有缓存
    else:
        word_vectors = None

    if cache is None:
        cache = EmbeddingCache(model, dataset)

    sim_scores = []
    human_scores = []
    oov_count = 0
    total_pairs = 0
    is_toefl = False
    with torch.no_grad():
        for item in test_data:
            total_pairs += 1
            # 安全解析测试项
            try:
                if len(item) == 3:
                    w1, w2, score = item
                else:
                    w1, w2 = item
                    score = 1.0
            except ValueError:
                print(f"无效测试项格式: {item}")
                continue
            if word_vectors:  # 从文件加载
                vec1 = word_vectors.get(w1)
                vec2 = word_vectors.get(w2)

                if vec1 is None or vec2 is None:
                    oov_count += 1
                    if CONFIG['oov_strategy'] == 'zero':
                        sim = 0.0
                    elif CONFIG['oov_strategy'] == 'random':
                        sim = np.random.uniform(0, 0.2)
                    else:
                        continue
                else:
                    if is_quantum:
                        rho1 = torch.tensor(vec1, dtype=torch.float32, device=device)
                        rho2 = torch.tensor(vec2, dtype=torch.float32, device=device)
                        sim = quantum_fidelity(rho1.unsqueeze(0), rho2.unsqueeze(0)).item()
                    else:
                        vec1 = torch.tensor(vec1, dtype=torch.float32, device=device)
                        vec2 = torch.tensor(vec2, dtype=torch.float32, device=device)
                        sim = nn.functional.cosine_similarity(vec1, vec2, dim=0).item()
            else:
            # 初始化默认值
                idx1 = idx2 = None

                try:
                    # 尝试获取词索引
                    idx1 = dataset.word2idx[w1]
                    idx2 = dataset.word2idx[w2]
                except KeyError as e:
                    oov_count += 1
                    # 处理OOV策略
                    if CONFIG['oov_strategy'] == 'zero':
                        sim = 0.0
                    elif CONFIG['oov_strategy'] == 'random':
                        sim = np.random.uniform(0, 0.2)
                    else:
                        continue
                else:
                    # 使用缓存获取嵌入
                    if is_quantum:
                        # 确保输入维度正确
                        L1 = model.emb(torch.tensor([[idx1]]).to(device))  # [1,1] -> [1,1,d,d]
                        L2 = model.emb(torch.tensor([[idx2]]).to(device))

                        rho1 = compute_density(L1).squeeze(0)  # [1,d,d] -> [d,d]
                        rho2 = compute_density(L2).squeeze(0)

                        # 添加维度检查
                        assert rho1.dim() == 2 and rho2.dim() == 2, f"维度错误: {rho1.shape}, {rho2.shape}"
                        sim = quantum_fidelity(rho1.unsqueeze(0), rho2.unsqueeze(0)).item()  # 转为3D输入
                    else:
                        # 经典模型使用缓存
                        vec1 = cache.get_embedding(w1)
                        vec2 = cache.get_embedding(w2)
                        if vec1 is None or vec2 is None:
                            continue
                        sim = nn.functional.cosine_similarity(vec1, vec2, dim=0).item()

            # 确保sim被定义
            if 'sim' in locals():
                sim_scores.append(sim)
                human_scores.append(score)

        valid_ratio = 1 - oov_count / total_pairs
        if is_toefl:
            # TOEFL准确率计算
            question_map = {}
            for (q, ans), sim in zip(test_data, sim_scores):
                if q not in question_map:
                    question_map[q] = []
                # 标记正确答案（假设A:开头的是正确选项）
                is_correct = 1.0 if ans.startswith('A:') else 0.0
                question_map[q].append((is_correct, sim))

            correct_count = 0
            total_questions = 0
            for q, answers in question_map.items():
                if len(answers) < 2:  # 至少需要1个正确选项和1个干扰项
                    continue
                sorted_ans = sorted(answers, key=lambda x: x[1], reverse=True)
                if sorted_ans[0][0] == 1.0:
                    correct_count += 1
                total_questions += 1
            if total_questions == 0:
                failure_reason = "无有效TOEFL题目"  # 新增失败原因
                return np.nan, valid_ratio, failure_reason  # 返回三个值
            accuracy = correct_count / total_questions
            return accuracy, valid_ratio, "成功"  # 返回三个值
        else:
            # 修正的缩进和逻辑
            if len(sim_scores) < 2:
                failure_reason = f"有效样本不足（{len(sim_scores)} < 2）"
                return np.nan, valid_ratio, failure_reason
            if len(set(sim_scores)) == 1:
                failure_reason = "所有相似度分数相同（无法计算相关性）"
                return np.nan, valid_ratio, failure_reason
            try:
                corr = spearmanr(sim_scores, human_scores).correlation
                failure_reason = "成功"  # 标记为成功
                return corr, valid_ratio, failure_reason
            except Exception as e:
                failure_reason = f"计算相关性时发生错误：{str(e)}"
                return np.nan, valid_ratio, failure_reason


def compute_von_neumann_entropy(cache):
    """计算所有词嵌入密度矩阵的平均冯诺依曼熵"""
    total_entropy = 0.0
    valid_count = 0
    epsilon = 1e-10  # 防止log(0)

    for word, entry in cache.cache.items():
        rho = torch.tensor(entry['embedding'], device=device)
        if rho.dim() == 2:
            rho = rho.unsqueeze(0)  # 添加batch维度

        # 确保密度矩阵合法
        rho = compute_density(rho) if rho.dim() == 4 else rho

        # 计算特征值
        eigvals = torch.linalg.eigvalsh(rho.squeeze(0))
        eigvals = eigvals.clamp(min=epsilon)  # 避免log(0)

        # 计算单个熵
        entropy = -torch.sum(eigvals * torch.log(eigvals))
        total_entropy += entropy.item()
        valid_count += 1  # 只统计有效词

    # 计算平均熵（防止除零）
    avg_entropy = total_entropy / valid_count if valid_count > 0 else 0.0
    return avg_entropy


def save_entropy_report(entropy, filename):
    """保存熵结果报告"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(f"Average Von Neumann Entropy: {entropy:.4f}\n")
    print(f"平均熵报告已保存至 {filename}")



def main():
    set_seed()

    # filename=["han","jin_nanbei","tang_wudai","song","yuan","ming","qing"]
    filename=["tang","wudai_song"]
    for f in filename:
        impath = f +"_segmented.txt"
        oppath = "xiuzheng/"+f +"_entropy.txt"
        dataset = QuantumTextDataset(impath, CONFIG['window_size'])
        dataloader = DataLoader(
            dataset,
            batch_size=CONFIG['batch_size'],
           collate_fn=collate_fn,  # 关键修改点
            shuffle=False
        )

        qmodel = QuantumCBOW(dataset.vocab_size, CONFIG['quantum_dim']).to(device)

        torch.cuda.empty_cache()  # 训练前清空缓存


        qoptim = optim.Adam(qmodel.parameters(), lr=CONFIG['lr'])
        print("Training Quantum Model...")
        train_model(qmodel, dataloader, qoptim, 'quantum')

        save_model(qmodel, qoptim,CONFIG['epochs'], 'quantum', is_best=True)

        qmodel, qcache = load_model(
            os.path.join(CONFIG['save_dir'], "quantum_best.pth"),
            dataset
        )

        print("\n计算平均冯诺依曼熵...")
        avg_entropy = compute_von_neumann_entropy(qcache)

        print(f"平均冯诺依曼熵: {avg_entropy:.4f}")
        save_entropy_report(avg_entropy, os.path.join(CONFIG['save_dir'], oppath))


if __name__ == "__main__":
    main()