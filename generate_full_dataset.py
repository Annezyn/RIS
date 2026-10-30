"""按 RIS/LIS 论文配置生成可直接训练的完整 DeepMIMO 数据集。

输出目录中的 X.npy 和 y.npy 可直接用于回归训练：X -> y。
"""

from pathlib import Path

import deepmimo as dm
import numpy as np
from tqdm.auto import tqdm


# ===================== 论文与数据集配置 =====================
SCENARIO = "o1_28"
RIS_BS = 4
UE_RX_SET = 0

# MATLAB 论文中的用户位置（使用 1 起始行、列编号）
GRID_COLUMNS = 181
FIXED_USER_ROW = 850
FIXED_USER_COLUMN = 90
UE_FIRST_ROW = 1000
UE_STOP_ROW = 1200  # 不包含 1200，即实际使用第 1000 至 1199 行

# RIS 与 OFDM 配置
RIS_SIDE = 32
ACTIVE_RIS_ELEMENTS = 8
TOTAL_SUBCARRIERS = 512
MODEL_SUBCARRIERS = 64
MAX_PATHS = 10
BANDWIDTH = 100e6

# 生成与划分配置
BATCH_SIZE = 100
VALIDATION_SIZE = 6200
RANDOM_SEED = 5
OUTPUT_DIR = "stm32_dataset"
# =============================================================


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / OUTPUT_DIR


def grid_index(row: int, column: int) -> int:
    """将 MATLAB 的 1 起始用户网格坐标变为 DeepMIMO 的 0 起始索引。"""
    return (row - 1) * GRID_COLUMNS + (column - 1)


def make_channel_parameters() -> dm.ChannelParameters:
    """创建新版 DeepMIMO 的 32 x 32 RIS 频域信道参数。"""
    parameters = dm.ChannelParameters()
    parameters.bs_antenna.shape = [RIS_SIDE, RIS_SIDE]
    parameters.bs_antenna.spacing = 0.5
    parameters.ue_antenna.shape = [1, 1]
    parameters.freq_domain = True
    parameters.num_paths = MAX_PATHS
    parameters.ofdm.bandwidth = BANDWIDTH
    parameters.ofdm.subcarriers = TOTAL_SUBCARRIERS
    parameters.ofdm.selected_subcarriers = np.arange(MODEL_SUBCARRIERS)
    parameters.ofdm.rx_filter = False
    return parameters


def make_ris_codebook(ris_side: int) -> tuple[np.ndarray, np.ndarray]:
    """复现论文的 1 x ris_side x ris_side UPA 反射码本。"""
    element_index = np.arange(ris_side)
    theta_y = np.deg2rad(20.0) - element_index * np.deg2rad(327.0 / ris_side)
    theta_z = element_index * np.pi / ris_side

    codebook_y = np.exp(-1j * element_index[:, None] * theta_y) / np.sqrt(ris_side)
    codebook_z = (
        np.exp(-1j * np.pi * element_index[:, None] * np.cos(theta_z))
        / np.sqrt(ris_side)
    )
    codebook = np.sqrt(ris_side * ris_side) * np.kron(codebook_z, codebook_y)
    beam_yz = np.column_stack(
        [
            np.tile(element_index, ris_side),
            np.repeat(element_index, ris_side),
        ]
    )
    return codebook.astype(np.complex64), beam_yz.astype(np.int16)


def load_hr(
    parameters: dm.ChannelParameters, user_batch: np.ndarray
) -> np.ndarray:
    """分批读取 RIS 到一批用户的完整频域信道，形状为 [B, 1024, 64]。"""
    dataset = dm.load(
        SCENARIO,
        tx_sets=[RIS_BS],
        rx_sets={UE_RX_SET: user_batch},
        max_paths=MAX_PATHS,
        compat_v3=True,
    )
    return dataset.compute_channels(parameters)[:, 0, :, :].conj()


def main() -> None:
    scenario_dir = PROJECT_DIR / "deepmimo_scenarios" / SCENARIO
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"未找到已解压的场景目录：{scenario_dir}")

    user_indices = np.arange(
        grid_index(UE_FIRST_ROW, 1),
        grid_index(UE_STOP_ROW, 1),
        dtype=np.int64,
    )
    n_users = len(user_indices)
    if n_users <= VALIDATION_SIZE:
        raise ValueError("用户数必须大于验证集大小。")

    OUTPUT_PATH.mkdir(exist_ok=True)
    parameters = make_channel_parameters()
    rng = np.random.default_rng(RANDOM_SEED)

    # Ht：固定用户位置到 RIS 的完整信道，形状 [1024, 64]。
    fixed_user_index = grid_index(FIXED_USER_ROW, FIXED_USER_COLUMN)
    fixed_dataset = dm.load(
        SCENARIO,
        tx_sets=[RIS_BS],
        rx_sets={UE_RX_SET: [fixed_user_index]},
        max_paths=MAX_PATHS,
        compat_v3=True,
    )
    ht = fixed_dataset.compute_channels(parameters)[0, 0, :, :]
    ris_codebook, beam_yz = make_ris_codebook(RIS_SIDE)

    print(f"将生成 {n_users:,} 个用户样本：训练 {n_users - VALIDATION_SIZE:,}，验证 {VALIDATION_SIZE:,}。")

    # 第一遍：依据完整 Ht、Hr 找到论文标签所需的全局幅值尺度。
    full_channel_scale = 0.0
    for start in tqdm(
        range(0, n_users, BATCH_SIZE),
        desc="1/3 扫描完整信道",
        unit="批",
    ):
        stop = min(start + BATCH_SIZE, n_users)
        hr = load_hr(parameters, user_indices[start:stop])
        full_channel_scale = max(
            full_channel_scale,
            float(np.abs(ht[None, :, :] * hr).max()),
        )

    if not np.isfinite(full_channel_scale) or full_channel_scale == 0:
        raise RuntimeError("完整级联信道无效，无法生成训练标签。")

    # 直接写入 .npy 内存映射文件，避免完整数据集占用大量内存。
    x_writer = np.lib.format.open_memmap(
        OUTPUT_PATH / "X.npy", mode="w+", dtype=np.float32, shape=(n_users, 1024)
    )
    y_writer = np.lib.format.open_memmap(
        OUTPUT_PATH / "y.npy", mode="w+", dtype=np.float32, shape=(n_users, 1024)
    )
    best_ris_index = np.empty(n_users, dtype=np.int16)

    sensor_indices = np.sort(
        rng.choice(RIS_SIDE * RIS_SIDE, ACTIVE_RIS_ELEMENTS, replace=False)
    )
    ht_active = ht[sensor_indices, :].T.reshape(-1)

    # 与论文一致的测量噪声参数。
    gt_db, gr_db, pt_db = 3, 3, 0
    nf_db, process_gain_db = 5, 10
    noise_power_db = (
        -204
        + 10 * np.log10(BANDWIDTH / TOTAL_SUBCARRIERS)
        + nf_db
        - process_gain_db
    )
    noise_power = 10 ** (0.1 * noise_power_db) / 10 ** (0.1 * (gt_db + gr_db + pt_db))
    noise_scale = np.sqrt(noise_power / 2)

    # 第二遍：由完整信道生成 y，并由 8 个有源单元生成未归一化 X。
    x_scale = 0.0
    for start in tqdm(
        range(0, n_users, BATCH_SIZE),
        desc="2/3 生成 X 与 y",
        unit="批",
    ):
        stop = min(start + BATCH_SIZE, n_users)
        batch_size = stop - start
        hr = load_hr(parameters, user_indices[start:stop])
        full_cascaded = ht[None, :, :] * hr

        # 论文使用第一个 OFDM 子载波评估全部 1024 个 RIS 码字。
        beam_response = full_cascaded[:, :, 0] @ ris_codebook
        relative_rate = np.log2(
            1 + (np.abs(beam_response) / full_channel_scale) ** 2
        )
        row_max_rate = relative_rate.max(axis=1, keepdims=True)
        if np.any(row_max_rate == 0):
            raise RuntimeError(f"样本 {start}:{stop} 的码字评分全为零。")
        labels = relative_rate / row_max_rate
        y_writer[start:stop] = labels.astype(np.float32)
        best_ris_index[start:stop] = labels.argmax(axis=1).astype(np.int16)

        # 只有 8 个有源单元的带噪级联信道是网络输入。
        hr_active = (
            hr[:, sensor_indices, :]
            .transpose(0, 2, 1)
            .reshape(batch_size, -1)
        )
        noise_1 = noise_scale * (
            rng.standard_normal((batch_size, ht_active.size), dtype=np.float32)
            + 1j * rng.standard_normal((batch_size, ht_active.size), dtype=np.float32)
        )
        noise_2 = noise_scale * (
            rng.standard_normal(hr_active.shape, dtype=np.float32)
            + 1j * rng.standard_normal(hr_active.shape, dtype=np.float32)
        )
        measured_channel = (ht_active[None, :] + noise_1) * (hr_active + noise_2)
        x_scale = max(x_scale, float(np.abs(measured_channel).max()))
        x_writer[start:stop] = np.stack(
            [measured_channel.real, measured_channel.imag], axis=-1
        ).reshape(batch_size, -1).astype(np.float32)

    if not np.isfinite(x_scale) or x_scale == 0:
        raise RuntimeError("模型输入归一化失败：测量信道无效。")
    x_writer.flush()
    y_writer.flush()

    # 第三遍：按整个数据集的最大值归一化 X，训练与 STM32 推理使用同一尺度。
    for start in tqdm(
        range(0, n_users, BATCH_SIZE),
        desc="3/3 归一化 X",
        unit="批",
    ):
        stop = min(start + BATCH_SIZE, n_users)
        x_writer[start:stop] /= x_scale
    x_writer.flush()

    # 论文采用随机训练/验证划分；30,000 用于训练，6,200 用于验证。
    permutation = np.random.default_rng(RANDOM_SEED).permutation(n_users)
    train_indices = permutation[:-VALIDATION_SIZE]
    validation_indices = permutation[-VALIDATION_SIZE:]
    best_ris_beam_yz = beam_yz[best_ris_index]

    # 保存训练、部署和复现实验所需的附加数据。
    np.save(OUTPUT_PATH / "Ht.npy", ht.astype(np.complex64))
    np.save(OUTPUT_PATH / "sensor_indices.npy", sensor_indices)
    np.save(OUTPUT_PATH / "user_indices.npy", user_indices)
    np.save(OUTPUT_PATH / "x_scale.npy", np.array([x_scale], dtype=np.float32))
    np.save(
        OUTPUT_PATH / "full_channel_scale.npy",
        np.array([full_channel_scale], dtype=np.float32),
    )
    np.save(OUTPUT_PATH / "ris_codebook.npy", ris_codebook)
    np.save(OUTPUT_PATH / "beam_yz.npy", beam_yz)
    np.save(OUTPUT_PATH / "best_ris_index.npy", best_ris_index)
    np.save(OUTPUT_PATH / "best_ris_beam_yz.npy", best_ris_beam_yz)
    np.save(OUTPUT_PATH / "train_indices.npy", train_indices)
    np.save(OUTPUT_PATH / "validation_indices.npy", validation_indices)

    print("\n完整数据集生成完成")
    print(f"X: {x_writer.shape}, {x_writer.dtype}")
    print(f"y: {y_writer.shape}, {y_writer.dtype}")
    print(f"训练样本: {len(train_indices):,}")
    print(f"验证样本: {len(validation_indices):,}")
    print(f"保存目录: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
