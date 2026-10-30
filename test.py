"""用 DeepMIMO O1_28 生成少量 STM32 模型输入样本。"""

import deepmimo as dm
import numpy as np
from pathlib import Path


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

    # 对应论文 UPA_codebook_generator.m 的 kron(F_CBz, F_CBy)。
    codebook = np.sqrt(ris_side * ris_side) * np.kron(codebook_z, codebook_y)
    beam_yz = np.column_stack(
        [
            np.tile(element_index, ris_side),
            np.repeat(element_index, ris_side),
        ]
    )
    return codebook.astype(np.complex64), beam_yz.astype(np.int16)


# ---------- 常用配置：调试时通常只需要修改 NUM_USERS ----------
SCENARIO = "o1_28"
NUM_USERS = 100
RIS_BS = 4
RIS_SIDE = 32
M = RIS_SIDE * RIS_SIDE
M_BAR = 8

K_TOTAL = 512
K_DL = 64
BW = 100e6

# MATLAB 论文代码中的位置，转换成 Python 的 0 起始索引
N_COLS = 181
TX_ROW, TX_COL = 850 - 1, 90 - 1
UE_START_ROW, UE_START_COL = 1000 - 1, 1 - 1

PROJECT_DIR = Path(__file__).resolve().parent
OUT = PROJECT_DIR / "stm32_dataset"
OUT.mkdir(exist_ok=True)

# 已验证 O1_28 的 UE 网格接收集编号为 0
UE_RX_SET = 0

# ---------- 用户网格索引 ----------
if not 1 <= NUM_USERS <= N_COLS - UE_START_COL:
    raise ValueError(f"NUM_USERS 必须在 1 到 {N_COLS - UE_START_COL} 之间。")

tx_user_index = TX_ROW * N_COLS + TX_COL
ue_start_index = UE_START_ROW * N_COLS + UE_START_COL
ue_indices = np.arange(ue_start_index, ue_start_index + NUM_USERS)

print(f"正在生成 {NUM_USERS} 个用户的模型输入……")

# ---------- 新版 DeepMIMO 信道参数 ----------
params = dm.ChannelParameters()
params.bs_antenna.shape = [RIS_SIDE, RIS_SIDE]
params.bs_antenna.spacing = 0.5
params.ue_antenna.shape = [1, 1]

params.freq_domain = True
params.num_paths = 10
params.ofdm.bandwidth = BW
params.ofdm.subcarriers = K_TOTAL
params.ofdm.selected_subcarriers = np.arange(K_DL)
params.ofdm.rx_filter = False

# ---------- Ht：固定发射端 → RIS ----------
ds_t = dm.load(
    SCENARIO,
    tx_sets=[RIS_BS],
    rx_sets={UE_RX_SET: [tx_user_index]},
    max_paths=10,
    compat_v3=True,
)

Ht = ds_t.compute_channels(params)[0, 0, :, :]
# Ht.shape = [1024, 64]

# ---------- 固定选择 8 个 RIS 单元，保证结果可复现 ----------
rng = np.random.default_rng(5)
sensor_indices = np.sort(rng.choice(M, size=M_BAR, replace=False))

# ---------- Hr：RIS → 每个用户 ----------
ds_r = dm.load(
    SCENARIO,
    tx_sets=[RIS_BS],
    rx_sets={UE_RX_SET: ue_indices},
    max_paths=10,
    compat_v3=True,
)

# 形状为 [用户数, 1024, 64]；论文中对这段信道取共轭
Hr = ds_r.compute_channels(params)[:, 0, :, :].conj()

# ---------- 由完整信道生成 RIS 标签 ----------
# 论文对每个用户评估完整 RIS 上的全部 32 x 32 = 1024 个反射码字，
# 以归一化速率向量作为回归标签；最大元素对应最佳 RIS 配置。
ris_codebook, beam_yz = make_ris_codebook(RIS_SIDE)
full_cascaded_channel = Ht[None, :, :] * Hr
full_channel_scale = float(np.abs(full_cascaded_channel).max())
if not np.isfinite(full_channel_scale) or full_channel_scale == 0:
    raise RuntimeError("完整级联信道无效，无法生成 RIS 标签。")

# 官方 MATLAB 代码使用第一个 OFDM 子载波为所有码字打分。
beam_response = full_cascaded_channel[:, :, 0] @ ris_codebook
relative_rate = np.log2(1 + (np.abs(beam_response) / full_channel_scale) ** 2)
row_max_rate = relative_rate.max(axis=1, keepdims=True)
if np.any(row_max_rate == 0):
    raise RuntimeError("码字评分全为零，无法生成 RIS 标签。")

y = (relative_rate / row_max_rate).astype(np.float32)
best_ris_index = y.argmax(axis=1).astype(np.int16)
best_ris_beam_yz = beam_yz[best_ris_index]

# ---------- 生成模型特征 ----------
# 论文中的噪声设置：发射、接收增益均为 3 dB，发射功率为 0 dB
GT_DB = 3
GR_DB = 3
PT_DB = 0
NF_DB = 5
PROCESS_GAIN_DB = 10

noise_power_db = -204 + 10 * np.log10(BW / K_TOTAL) + NF_DB - PROCESS_GAIN_DB
noise_power_bar = 10 ** (0.1 * noise_power_db) / 10 ** (0.1 * (GT_DB + GR_DB + PT_DB))

# 只取有源 RIS 单元；每名用户得到 8 × 64 = 512 个复数特征
ht_bar = Ht[sensor_indices, :].T.reshape(-1)
hr_bar = Hr[:, sensor_indices, :].transpose(0, 2, 1).reshape(NUM_USERS, -1)

noise_1 = np.sqrt(noise_power_bar / 2) * (
    rng.standard_normal(ht_bar.shape) + 1j * rng.standard_normal(ht_bar.shape)
)
noise_2 = np.sqrt(noise_power_bar / 2) * (
    rng.standard_normal(hr_bar.shape) + 1j * rng.standard_normal(hr_bar.shape)
)

# 级联信道，并把每个复数拆成实部和虚部；输出形状为 [用户数, 1024]
z = (ht_bar[None, :] + noise_1[None, :]) * (hr_bar + noise_2)
max_abs = float(np.abs(z).max())
if not np.isfinite(max_abs) or max_abs == 0:
    raise RuntimeError("信道归一化失败：最大幅值不是有效正数。")

X = np.stack([z.real, z.imag], axis=-1).reshape(NUM_USERS, -1).astype(np.float32)
X /= max_abs

# ---------- 保存模型输入及复现实验需要的信息 ----------
np.save(OUT / "X.npy", X)
np.save(OUT / "y.npy", y)
np.save(OUT / "Ht.npy", Ht.astype(np.complex64))
np.save(OUT / "sensor_indices.npy", sensor_indices)
np.save(OUT / "user_indices.npy", ue_indices)
np.save(OUT / "x_scale.npy", np.array([max_abs], dtype=np.float32))
np.save(OUT / "full_channel_scale.npy", np.array([full_channel_scale], dtype=np.float32))
np.save(OUT / "best_ris_index.npy", best_ris_index)
np.save(OUT / "best_ris_beam_yz.npy", best_ris_beam_yz)
np.save(OUT / "ris_codebook.npy", ris_codebook)
np.save(OUT / "beam_yz.npy", beam_yz)

print("生成完成")
print("X 的形状:", X.shape)
print("X 的数据类型:", X.dtype)
print("y 的形状:", y.shape)
print("最佳 RIS 码字索引:", best_ris_index)
print("保存目录：", OUT.resolve())
