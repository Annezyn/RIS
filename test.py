from pathlib import Path
import numpy as np
import deepmimo as dm

# ---------- 论文配置 ----------
SCENARIO = "o1_28"
RIS_BS = 4
RIS_SIDE = 32
M = RIS_SIDE * RIS_SIDE
M_BAR = 8

K_TOTAL = 512
K_DL = 64
BW = 100e6

# MATLAB 论文代码中的位置，转换为 Python 的 0 起始索引
N_COLS = 181
TX_ROW, TX_COL = 850 - 1, 90 - 1
UE_ROWS = range(1000 - 1, 1200 - 1)  # 共 200 行，即 36,200 个用户

OUT = Path("stm32_dataset")
OUT.mkdir(exist_ok=True)

# ---------- 下载并检查场景 ----------
scenario_dir = Path.cwd() / "deepmimo_scenarios" / SCENARIO
if not scenario_dir.is_dir():
    dm.download(SCENARIO)

if not scenario_dir.is_dir():
    raise RuntimeError(
        "场景下载未完成，脚本已停止。请等待 DeepMIMO 下载服务恢复后重试；"
        "不要继续执行信道生成步骤。"
    )

# O1 中通常只有一个 UE 网格接收集；打印后可核对编号
sets = dm.get_txrx_sets(SCENARIO)
for item in sets:
    print(item)

# 若打印结果中的 UE 网格编号不是 0，修改这里
UE_RX_SET = 0

# ---------- 用户网格索引 ----------
tx_user_index = TX_ROW * N_COLS + TX_COL

ue_indices = np.concatenate([
    np.arange(row * N_COLS, (row + 1) * N_COLS)
    for row in UE_ROWS
])

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

# ---------- 只随机选一次感知单元，并保存 ----------
rng = np.random.default_rng(5)
sensor_indices = np.sort(rng.choice(M, size=M_BAR, replace=False))

np.save(OUT / "Ht.npy", Ht.astype(np.complex64))
np.save(OUT / "sensor_indices.npy", sensor_indices)

# ---------- Hr：RIS → 每个用户 ----------
ds_r = dm.load(
    SCENARIO,
    tx_sets=[RIS_BS],
    rx_sets={UE_RX_SET: ue_indices},
    max_paths=10,
    compat_v3=True,
)

n_users = len(ue_indices)
feature_dim = 2 * M_BAR * K_DL
X = np.empty((n_users, feature_dim), dtype=np.float32)

# 论文中的噪声设置
GT_DB = 3
GR_DB = 3
PT_DB = 0
NF_DB = 5
PROCESS_GAIN_DB = 10

noise_power_db = -204 + 10 * np.log10(BW / K_TOTAL) + NF_DB - PROCESS_GAIN_DB
noise_power_bar = 10 ** (0.1 * noise_power_db) / 10 ** (0.1 * (GT_DB + GR_DB + PT_DB))

# Ht 中只取有源 RIS 单元；顺序与 MATLAB reshape 接近
ht_bar = Ht[sensor_indices, :].T.reshape(-1)

batch_size = 100
max_abs = 0.0

for start in range(0, n_users, batch_size):
    stop = min(start + batch_size, n_users)

    batch = ds_r.trim(idxs=np.arange(start, stop))
    h = batch.compute_channels(params)[:, 0, :, :]

    # 作者代码对 RIS→UE 信道取共轭
    hr = h.conj()

    # [batch, M_BAR, K_DL] → [batch, K_DL, M_BAR] → [batch, M_BAR*K_DL]
    hr_bar = hr[:, sensor_indices, :].transpose(0, 2, 1).reshape(stop - start, -1)

    noise_1 = np.sqrt(noise_power_bar / 2) * (
        rng.standard_normal(ht_bar.shape) + 1j * rng.standard_normal(ht_bar.shape)
    )

    noise_2 = np.sqrt(noise_power_bar / 2) * (
        rng.standard_normal(hr_bar.shape) + 1j * rng.standard_normal(hr_bar.shape)
    )

    z = (ht_bar[None, :] + noise_1[None, :]) * (hr_bar + noise_2)

    # 每个复数特征变为 [Re, Im]；最终维度为 2 × M_BAR × K_DL
    X[start:stop] = np.stack([z.real, z.imag], axis=-1).reshape(stop - start, -1)
    max_abs = max(max_abs, float(np.abs(z).max()))

    print(f"完成 {stop}/{n_users}")

# 论文使用整个数据集的最大级联信道幅度进行缩放
X /= max_abs

np.save(OUT / "X.npy", X)
np.save(OUT / "x_scale.npy", np.array([max_abs], dtype=np.float32))

print("X:", X.shape)
print("保存目录：", OUT.resolve())
