"""RIS codebook-selection MLP training and inference.

The model maps a 1024-dimensional sampled CSI feature vector to 1024
normalized achievable-rate scores. Inference selects the codebook with
``argmax`` over the predicted scores. This is a regression task.

RIS 码本选择任务的 MLP 训练与推理模块。

任务输入:
    X.npy 的每一行是一条样本，形状为 [1024]；对应 8 个有源 RIS 单元、
    64 个 OFDM 子载波上的带噪采样级联 CSI，复数的实部、虚部拼接后
    形成 8 * 64 * 2 = 1024 个实数特征。

回归标签:
    Y.npy 的形状为 [N, 1024]，每一行表示使用 1024 个 RIS 码本时的
    归一化可达速率评分，数值范围约为 0~1；码本由 32 x 32 DFT 波束
    组成。模型输出仍是连续评分，不经过 softmax。

推理方式:
    对模型输出的 1024 个评分执行 argmax，得到最优 RIS 码本编号。

评估指标:
    avg_rate (R_plot)，即预测速率向量最大值的样本均值；genie-aided
    上界为真实 Y 速率向量最大值的样本均值。
    注意：当前数据集标签、avg_rate 和 genie_rate 均仅使用 OFDM 子载波 0，
    不是论文中全部子载波的平均速率，和论文完整指标存在差异。

数据集来源:
    DeepMIMO O1_28 射线追踪场景；训练集、验证集和独立测试集划分来自
    外部索引文件，本模块不会重新随机切分数据。

通信术语:
    级联信道: BS -> RIS 与 UE -> RIS 两段信道逐单元相乘，描述信号经
        RIS 反射后到达 UE 的有效路径。
    码本: 预先设计的 1024 组 RIS 相位配置，每组配置对应一个反射波束。
    子载波: OFDM 频域信道采样点；每条输入样本使用 64 个子载波。
"""

# ============================================================================
# 模块 01：导入依赖
# 汇总训练、评估、推理和命令行解析所需的 Python 与 PyTorch 依赖。
# ============================================================================

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# 模块 02：全局配置与文件常量
# 统一定义默认数据文件、模型尺寸、训练超参数和输出日志字段。
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
# 默认权重文件放在脚本所在目录，便于训练和推理共用同一路径。
DEFAULT_WEIGHT_PATH = SCRIPT_DIR / "ris_mlp_weight.pth"
DEFAULT_LOG_PATH = SCRIPT_DIR / "train_log.csv"
TRAIN_LOG_FIELDNAMES = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_avg_rate",
    "test_avg_rate",
    "test_genie_rate",
)

# 训练所需的外部数据文件：输入、标签、训练、验证和独立测试索引。
DATA_FILENAMES = (
    "X.npy",
    "Y.npy",
    "train_indices.npy",
    "val_indices.npy",
    "test_indices.npy",
)

# X 每条样本为 1024 维；同时对应 1024 个候选 RIS 码本。
INPUT_DIM = 1024
OUTPUT_DIM = 1024

# 默认训练超参数对齐论文训练器。
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 500
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_SGD_MOMENTUM = 0.0
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_SCALE_FACTOR = 1
DEFAULT_HIDDEN_LAYERS = 3
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_NUM_WORKERS = 0
DEFAULT_SEED = 42

# 论文 ReduceLROnPlateau 与 early stopping 配置。
LR_SCHEDULER_FACTOR = 0.75
LR_SCHEDULER_PATIENCE = 4
LR_SCHEDULER_THRESHOLD = 1e-3
LR_SCHEDULER_MIN_LR = 1e-4
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 5e-4


# ============================================================================
# 模块 03：运行环境与数据路径辅助
# 负责定位真实数据集目录、固定随机种子并选择 GPU 或 CPU。
# ============================================================================

def resolve_data_dir(explicit_dir: str | None = None) -> Path:
    """Find the directory containing all required dataset files.

    The explicit path has highest priority. Otherwise, check the current
    working directory, this script's directory, and its parent directory.
    This makes the script work both inside the repository and from Jupyter.
    """
    # 优先使用命令行显式给出的数据目录，避免误读其他位置的旧数据。
    if explicit_dir is not None:
        candidates = [Path(explicit_dir).expanduser().resolve()]
    else:
        # 默认依次检查当前目录、脚本目录和脚本上级目录。
        candidates = [
            Path.cwd(),
            SCRIPT_DIR,
            SCRIPT_DIR.parent,
        ]

    # 记录已检查路径，避免重复检查；字段类型为 list[Path]。
    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        # 全部数据文件同时存在时，才认为该目录是有效数据集目录。
        if all((candidate / name).is_file() for name in DATA_FILENAMES):
            return candidate

    # 找不到完整数据集时给出已检查路径，便于定位路径配置问题。
    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not find X.npy, Y.npy, train_indices.npy, "
        f"val_indices.npy, and test_indices.npy. Checked:\n{checked_text}\n"
        "Pass the correct directory with --data-dir."
    )


def set_random_seed(seed: int) -> None:
    """Seed libraries used by training and DataLoader shuffling only."""
    # 这里只控制训练随机性；训练/验证 split 始终来自外部索引文件。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Use CUDA when available; otherwise fall back to CPU."""
    # 优先使用 CUDA GPU；没有可用 GPU 时自动退回 CPU。
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 模块 04：RIS 数据集类
# 使用外部索引按需读取带噪级联 CSI 和对应的码本速率评分。
# ============================================================================

class RISRegressionDataset(Dataset):
    """Map provided split indices to on-demand samples from memory maps.

    数据视图:
        x_data: [N, 1024]，每条样本为带噪采样级联 CSI 特征。
        y_data: [N, 1024]，每条样本为 1024 个码本的归一化速率评分。
        indices: [N_split]，外部训练、验证或测试划分中的原始样本行号。

    由于数组通过 mmap 只读打开，完整数据集不会一次性复制到内存或显存。
    """

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        """保存数据视图和外部划分索引，并检查形状与索引范围。

        Args:
            x_data: [N, 1024] 输入特征数组。
            y_data: [N, 1024] 回归标签数组。
            indices: [N_split] 当前数据集使用的样本行索引。
        """
        if x_data.ndim != 2 or y_data.ndim != 2:
            raise ValueError("X.npy and Y.npy must both be two-dimensional.")
        if x_data.shape != y_data.shape:
            raise ValueError(
                f"X and Y shapes differ: {x_data.shape} vs {y_data.shape}."
            )

        # 保存的是 mmap/ndarray 引用，不会在此处加载全部样本。
        self.x_data = x_data
        self.y_data = y_data
        # flatten 后形状为 [N_split]；训练/验证索引不会被重新生成。
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)

        if self.indices.size == 0:
            raise ValueError("The provided index array is empty.")
        if self.indices.min() < 0 or self.indices.max() >= x_data.shape[0]:
            raise IndexError("A split index is outside the X/Y sample range.")

    def __len__(self) -> int:
        """返回当前训练或验证划分包含的样本数。"""
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按外部索引读取一条样本。

        Returns:
            x_sample: [1024]，一条 RIS 采样 CSI 特征。
            y_sample: [1024]，对应 1024 个码本的归一化速率评分。
        """
        sample_index = int(self.indices[item])

        # Copy only the current row, not the complete dataset.
        # 复制当前行后形状仍为 [1024]，这样一个 batch 的形状是 [B, 1024]。
        x_sample = np.array(
            self.x_data[sample_index],
            dtype=np.float32,
            copy=True,
        )
        # y_sample 形状为 [1024]；每个元素对应一个 RIS 码本的评分。
        y_sample = np.array(
            self.y_data[sample_index],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(x_sample), torch.from_numpy(y_sample)


# ============================================================================
# 模块 05：MLP 模型定义
# 按论文宽度公式构建从 CSI 特征到 1024 个码本评分的纯 MLP 回归器。
# ============================================================================

class RISCodebookMLP(nn.Module):
    """Paper-style fully connected RIS achievable-rate regressor.

    输入层接收 1024 维采样 CSI 特征，输出层产生 1024 个连续速率评分。
    每个输出维度对应一个 RIS 码本；输出层不接 softmax 或 sigmoid。

    隐藏层宽度遵循论文公式 [IF/G, 4*IF/G, 4*IF/G]，其中 IF=1024。
    hl 指定使用的隐藏层数量，支持 0、1、2、3；每个隐藏层后只接 ReLU。
    """

    def __init__(
        self,
        hidden_layers: int = 3,
        scale_factor: int = 1,
    ) -> None:
        """按论文宽度公式构建纯 MLP，不使用 Dropout。"""
        super().__init__()
        if scale_factor <= 0:
            raise ValueError("--G must be a positive integer.")
        if not 0 <= hidden_layers <= 3:
            raise ValueError("--hl must be between 0 and 3.")

        paper_widths = (
            INPUT_DIM // scale_factor,
            (4 * INPUT_DIM) // scale_factor,
            (4 * INPUT_DIM) // scale_factor,
        )
        if hidden_layers > 0 and paper_widths[0] <= 0:
            raise ValueError("--G is too large for the 1024-dimensional input.")

        self.hidden_dims = paper_widths[:hidden_layers]
        layers: list[nn.Module] = []
        previous_dim = INPUT_DIM
        for hidden_dim in self.hidden_dims:
            # [B, D_in] -> [B, D_hidden]
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim

        # No activation: [B, D_last] -> [B, 1024] continuous rate scores.
        layers.append(nn.Linear(previous_dim, OUTPUT_DIM))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行前向传播。

        Args:
            x: [B, 1024] 的 CSI 特征 batch。

        Returns:
            scores: [B, 1024]，每条样本对应 1024 个 RIS 码本评分。
        """
        return self.network(x)


# ============================================================================
# 模块 06：数据加载模块
# 按给定训练、验证和测试索引构建按需加载的 DataLoader。
# ============================================================================

def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build loaders using exactly the provided train/validation/test indices.

    返回:
        train_loader: 训练 DataLoader，每个 batch 形状 [B, 1024]。
        validation_loader: 验证 DataLoader，每个 batch 形状 [B, 1024]。
        test_loader: 独立测试 DataLoader，每个 batch 形状 [B, 1024]。
        train_indices: [N_train]，外部训练索引。
        validation_indices: [N_val]，外部验证索引。
        test_indices: [N_test]，外部独立测试索引。

    DataLoader 的 shuffle 只改变 batch 顺序，不会重新划分数据集。
    """
    x_path = data_dir / "X.npy"
    y_path = data_dir / "Y.npy"

    # Read-only memory maps keep the full arrays off the Python heap and GPU.
    # x_data: [N, 1024]；y_data: [N, 1024]。
    x_data = np.load(x_path, mmap_mode="r")
    y_data = np.load(y_path, mmap_mode="r")
    # X 已由数据生成器按全局 x_scale 归一化，Y 的每行已按自身最大值归一化。
    # 本训练模块不再重复缩放，避免改变原始 CSI 相对幅度和标签排序。
    # train_indices: [N_train]；validation_indices: [N_val]；test_indices: [N_test]。
    train_indices = np.load(data_dir / "train_indices.npy")
    validation_indices = np.load(data_dir / "val_indices.npy")
    test_indices = np.load(data_dir / "test_indices.npy")

    # 特征维度和码本评分维度分别为 1024；不一致时立即报错。
    if x_data.shape[1] != INPUT_DIM or y_data.shape[1] != OUTPUT_DIM:
        raise ValueError(
            "Expected X/Y feature dimensions of "
            f"{INPUT_DIM}/{OUTPUT_DIM}, got "
            f"{x_data.shape[1]}/{y_data.shape[1]}."
        )
    # 训练集、验证集和独立测试集禁止重叠，防止数据泄漏。
    if (
        np.intersect1d(train_indices, validation_indices).size
        or np.intersect1d(train_indices, test_indices).size
        or np.intersect1d(validation_indices, test_indices).size
    ):
        raise ValueError("Training, validation, and test indices overlap.")

    # 三个 Dataset 共享同一组 mmap 数组，但使用不同的外部索引。
    train_dataset = RISRegressionDataset(x_data, y_data, train_indices)
    validation_dataset = RISRegressionDataset(
        x_data,
        y_data,
        validation_indices,
    )
    test_dataset = RISRegressionDataset(x_data, y_data, test_indices)

    loader_options = {
        # num_workers=0 在 Windows/Jupyter 下最稳妥。
        "num_workers": num_workers,
        # CUDA 训练时启用锁页内存，有助于加速主机到 GPU 的拷贝。
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        # 多进程读取时保留 worker，并预取 2 个 batch。
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 2

    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle batches only; never perform a new split.
        # 仅打乱已有训练索引的顺序，不改变训练/验证/测试集组成。
        generator=shuffle_generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )
    return (
        train_loader,
        validation_loader,
        test_loader,
        train_indices,
        validation_indices,
        test_indices,
    )


# ============================================================================
# 模块 07：单轮训练循环
# 完成一个 epoch 的前向传播、MSE 损失计算、反向传播和参数更新。
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """训练一个完整 epoch，返回按样本数加权的平均 MSE。

    batch 维度:
        x_batch: [B, 1024]
        y_batch: [B, 1024]
        predictions: [B, 1024]
        loss: 标量
    """
    # 训练模式会启用 Dropout。
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        # 将 [B, 1024] 的输入和标签移动到 GPU 或 CPU。
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # 清空上一轮梯度，避免梯度累加。
        optimizer.zero_grad(set_to_none=True)
        # 回归前向：predictions 的形状为 [B, 1024]。
        predictions = model(x_batch)
        # MSE 对 [B, 1024] 中所有元素求平方误差均值，返回标量损失。
        loss = criterion(predictions, y_batch)
        # 对 1024 个码本评分共同回传梯度。
        loss.backward()
        # 限制梯度的全局范数，降低训练不稳定或梯度爆炸的风险。
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        # 使用 SGD 更新网络参数。
        optimizer.step()

        # 按当前 batch 样本数加权，最后得到整个 epoch 的平均损失。
        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ============================================================================
# 模块 08：评估指标计算
# 统一计算验证集或测试集 loss、R_plot 和 genie-aided 速率。
# ============================================================================

# 当前 avg_rate 和 genie_rate 基于仅使用 OFDM 子载波 0 的数据集标签计算。
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Return loss, predicted average rate, and genie rate.

    论文式 (13) 的 R_plot:
        对每个样本的预测速率向量取最大预测速率，再对数据划分内所有
        样本求均值。

    genie-aided 上界:
        对每个样本的真实 Y 速率向量取最大速率，再对数据划分内所有
        样本求均值。本地 Y 是归一化速率，因此指标也位于 0~1。
    """
    # 验证模式会关闭 Dropout，并使用当前参数做确定性推理。
    model.eval()
    total_loss = 0.0
    total_avg_rate = 0.0
    total_genie_rate = 0.0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        # x_batch: [B, 1024]；y_batch: [B, 1024]。
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # predictions: [B, 1024]；loss 为标量 MSE。
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)

        # predicted_rates: [B]；genie_rates: [B]。
        predicted_rates = predictions.max(dim=1).values
        genie_rates = y_batch.max(dim=1).values

        # 按样本数累加损失、预测速率和 genie 上界速率。
        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_avg_rate += predicted_rates.sum().item()
        total_genie_rate += genie_rates.sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_avg_rate / total_samples,
        total_genie_rate / total_samples,
    )


# ============================================================================
# 模块 09：权重加载与推理辅助
# 重建训练时的模型结构、加载权重并从单条 CSI 特征预测最优码本编号。
# ============================================================================

def load_trained_model(
    weight_path: str | Path = DEFAULT_WEIGHT_PATH,
    device: torch.device | None = None,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
) -> tuple[RISCodebookMLP, torch.device]:
    """Reconstruct the model and load saved weights for inference.

    Args:
        weight_path: 保存 state_dict 的 .pth 文件。
        device: 推理设备；None 时自动选择 CUDA/CPU。
        scale_factor: 论文宽度公式中的缩放因子 G。
        hidden_layers: 隐藏层数量 hl。

    Returns:
        model: 已加载权重并处于 eval 模式的 MLP。
        device: 实际使用的 PyTorch 设备。
    """
    weight_path = Path(weight_path)
    if not weight_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weight_path}")

    if device is None:
        device = select_device()

    # 网络结构必须与训练时一致，state_dict 才能正确加载。
    model = RISCodebookMLP(
        scale_factor=scale_factor,
        hidden_layers=hidden_layers,
    ).to(device)
    # state_dict 包含各 Linear 层的参数张量。
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    # 进入确定性推理模式。
    model.eval()
    return model, device


@torch.no_grad()
def predict_best_codebook(
    model: RISCodebookMLP,
    feature: np.ndarray | torch.Tensor,
    device: torch.device,
) -> int:
    """Return the best RIS codebook index for one 1024-dimensional feature.

    Args:
        model: 已训练的 RISCodebookMLP。
        feature: [1024] 的单条采样 CSI 特征。
        device: 推理设备。

    Returns:
        RIS 码本编号，范围为 0~1023。
    """
    if isinstance(feature, np.ndarray):
        # NumPy 特征 [1024] -> CPU Tensor [1024]。
        feature_tensor = torch.from_numpy(
            np.array(feature, dtype=np.float32, copy=True)
        )
    elif isinstance(feature, torch.Tensor):
        # 传入 Tensor 时转成 CPU float32；后续再统一移动到推理设备。
        feature_tensor = feature.detach().to(dtype=torch.float32, device="cpu")
    else:
        raise TypeError("feature must be a NumPy array or a PyTorch tensor.")

    # 无论原输入是 [1024]、[1, 1024] 还是其他单样本布局，都展平为 [1024]。
    feature_tensor = feature_tensor.reshape(-1)
    # 必须保证最终输入特征数恰好为 1024。
    if feature_tensor.numel() != INPUT_DIM:
        raise ValueError(
            f"Expected {INPUT_DIM} input features, "
            f"got {feature_tensor.numel()}."
        )

    # 推理阶段使用 eval 模式。
    model.eval()
    # [1024] -> [1, 1024]；scores 形状为 [1, 1024]。
    scores = model(feature_tensor.unsqueeze(0).to(device))
    # argmax(dim=1) 得到 [1]，取标量并转换为 Python int 码本编号。
    return int(scores.argmax(dim=1).item())


# ============================================================================
# 模块 10：完整训练流程
# 组织数据加载、训练、验证、学习率调度、早停、最优权重保存和最终测试。
# ============================================================================

def train_mlp(args: argparse.Namespace) -> Path:
    """执行完整训练、验证、最佳权重保存和权重恢复。

    Args:
        args: 命令行参数集合，包含数据目录、epoch、batch、学习率、
            G、hl、梯度裁剪阈值等。

    Returns:
        最佳验证集 loss 权重文件路径。
    """
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.G <= 0:
        raise ValueError("--G must be a positive integer.")
    if not 0 <= args.hl <= 3:
        raise ValueError("--hl must be between 0 and 3.")

    # 固定随机种子，使参数初始化和训练 batch shuffle 可复现。
    set_random_seed(args.seed)
    # 优先使用 GPU；无 GPU 时自动使用 CPU。
    device = select_device()
    # 解析包含训练、验证和独立测试索引的数据目录。
    data_dir = resolve_data_dir(args.data_dir)
    # 权重保存路径；必要时创建父目录。
    weight_path = Path(args.weights).expanduser().resolve()
    weight_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    (
        train_loader,
        validation_loader,
        test_loader,
        train_indices,
        validation_indices,
        test_indices,
    ) = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )
    # test_loader is used only after training and best-checkpoint loading.
    # train_indices: [N_train]；validation_indices: [N_val]。
    print(f"Training samples: {len(train_indices):,}")
    print(f"Validation samples: {len(validation_indices):,}")
    print(f"Test samples: {len(test_indices):,}")

    # 模型输入 [B, 1024]，输出 [B, 1024]；宽度由论文 G 和 hl 决定。
    model = RISCodebookMLP(
        hidden_layers=args.hl,
        scale_factor=args.G,
    ).to(device)
    print(
        f"MLP hidden dimensions (HL={args.hl}, G={args.G}): "
        f"{list(model.hidden_dims)}"
    )
    # MSE 对 batch 内所有码本评分计算均方误差。
    criterion = nn.MSELoss()
    # 这是论文对应的 SGD 优化器；此前基线使用的 Adam 优化器在这里明确区分。
    # 初始参数 lr=0.01、momentum=0.9、weight_decay=1e-4 均可由命令行覆盖。
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    print(
        "Optimizer: SGD | "
        f"lr={args.learning_rate:g} | "
        f"momentum={args.momentum:g} | "
        f"weight_decay={args.weight_decay:g}"
    )
    # 调度器监控验证集 loss，按论文参数逐步降低学习率。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        threshold=LR_SCHEDULER_THRESHOLD,
        threshold_mode="abs",
        min_lr=LR_SCHEDULER_MIN_LR,
    )

    best_validation_loss = float("inf")
    early_stopping_reference_loss = float("inf")
    epochs_without_improvement = 0
    log_path = DEFAULT_LOG_PATH
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        log_writer = csv.DictWriter(log_file, fieldnames=TRAIN_LOG_FIELDNAMES)
        log_writer.writeheader()
    print(f"Training log: {log_path}")

    for epoch in range(1, args.epochs + 1):
        # 跑完一个 epoch 的全部训练 batch，返回平均 train MSE。
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip,
        )
        # 验证集计算 MSE、论文 R_plot 预测平均速率和 genie 上界速率。
        validation_loss, validation_avg_rate, _ = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        # 每轮输出训练损失、验证损失和验证平均速率。
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={validation_loss:.6f} | "
            f"val_avg_rate={validation_avg_rate:.6f}"
        )

        with log_path.open("a", newline="", encoding="utf-8") as log_file:
            log_writer = csv.DictWriter(
                log_file,
                fieldnames=TRAIN_LOG_FIELDNAMES,
            )
            log_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{validation_loss:.6f}",
                    "val_avg_rate": f"{validation_avg_rate:.6f}",
                }
            )

        # 只要验证 loss 创新低就保存当前最优模型。
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            # 仅保存模型参数 state_dict，不保存优化器和数据。
            torch.save(model.state_dict(), weight_path)

        # 早停只计算达到最小改进幅度的验证 loss 改善。
        if (
            validation_loss
            < early_stopping_reference_loss - EARLY_STOPPING_MIN_DELTA
        ):
            early_stopping_reference_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # 调度器步进必须在验证 loss 计算完成之后。
        scheduler.step(validation_loss)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping: validation loss did not improve by at least "
                f"{EARLY_STOPPING_MIN_DELTA:g} for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    # Reload the best validation checkpoint so subsequent inference is consistent.
    # 将内存中的模型恢复为验证集 loss 最优权重。
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Final validation avg_rate: {validation_avg_rate:.6f}")

    # Evaluate test only once, after training and best-checkpoint loading.
    _, test_avg_rate, test_genie_rate = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )
    print(
        f"test_avg_rate={test_avg_rate:.6f} | "
        f"test_genie_rate={test_genie_rate:.6f}"
    )

    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        log_writer = csv.DictWriter(log_file, fieldnames=TRAIN_LOG_FIELDNAMES)
        log_writer.writerow(
            {
                "epoch": "test",
                "test_avg_rate": f"{test_avg_rate:.6f}",
                "test_genie_rate": f"{test_genie_rate:.6f}",
            }
        )
    return weight_path


# ============================================================================
# 模块 11：单样本推理示例
# 从真实数据中读取一条 CSI 特征，调用训练好的模型输出最优 RIS 码本编号。
# ============================================================================

def inference_example(
    data_dir: str | Path | None = None,
    weight_path: str | Path = DEFAULT_WEIGHT_PATH,
    sample_index: int = 0,
    feature: np.ndarray | torch.Tensor | None = None,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
) -> int:
    """Predict one codebook index from a supplied or file-loaded feature.

    如果未传入 feature，则从 X.npy 读取 sample_index 对应的一行；
    如果传入 feature，则完全使用内存中的单条 [1024] 特征。
    """
    if feature is None:
        # 回退到磁盘读取：features 形状为 [N, 1024]。
        resolved_data_dir = resolve_data_dir(
            None if data_dir is None else str(data_dir)
        )
        features = np.load(
            resolved_data_dir / "X.npy",
            mmap_mode="r",
        )
        # 读取单行并复制为 [1024] 的内存数组。
        feature = np.array(features[sample_index], dtype=np.float32, copy=True)

    # 重建网络、加载权重并自动选择 CUDA/CPU。
    model, device = load_trained_model(
        weight_path,
        device=select_device(),
        scale_factor=scale_factor,
        hidden_layers=hidden_layers,
    )
    # 对单条特征 [1024] 执行 argmax，得到 RIS 码本编号。
    best_index = predict_best_codebook(model, feature, device)
    print(f"Sample {sample_index}: best RIS codebook index = {best_index}")
    return best_index


# ============================================================================
# 模块 12：命令行参数解析
# 集中定义训练、模型缩放、推理和复现实验所需的命令行参数。
# ============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数，并返回 argparse.Namespace。"""
    parser = argparse.ArgumentParser(
        description="Train a lightweight MLP for RIS codebook selection."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Directory containing X.npy, Y.npy, and train/validation/test "
            "split indices."
        ),
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHT_PATH),
        help="Output path for ris_mlp_weight.pth.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=DEFAULT_SGD_MOMENTUM,
        help="Momentum for the paper-aligned SGD optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay (L2 regularization) for the SGD optimizer.",
    )
    parser.add_argument(
        "--G",
        type=int,
        default=DEFAULT_SCALE_FACTOR,
        help="Paper width scaling factor G.",
    )
    parser.add_argument(
        "--hl",
        type=int,
        default=DEFAULT_HIDDEN_LAYERS,
        help="Number of hidden layers (0 to 3).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Keep 0 on Windows/Jupyter unless worker processes are needed.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Skip training and run the single-sample inference example.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="X.npy row used by the inference example.",
    )
    return parser.parse_args()


# ============================================================================
# 模块 13：程序主入口
# 根据是否传入 --inference 选择单样本推理或完整训练流程。
# ============================================================================

if __name__ == "__main__":
    # 直接运行脚本时：默认训练；指定 --inference 时只执行单样本推理。
    cli_args = parse_args()
    if cli_args.inference:
        # 从 X.npy 读取指定样本并输出最优 RIS 码本编号。
        inference_example(
            data_dir=cli_args.data_dir,
            weight_path=cli_args.weights,
            sample_index=cli_args.sample_index,
            scale_factor=cli_args.G,
            hidden_layers=cli_args.hl,
        )
    else:
        # 使用磁盘真实数据和外部索引完整训练 MLP。
        train_mlp(cli_args)
