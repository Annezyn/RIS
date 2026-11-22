"""生成 RIS 数据：X 取前 64 个子载波，Y 对完整 512 个子载波的速率取平均。"""

from pathlib import Path

import deepmimo as dm
import numpy as np
from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "stm32_dataset_512"
SCENARIO = "o1_28"
RIS_BS = 4  # 本地新版场景的 tx_set 4 对应论文旧版 BS 3（TX_3）。
UE_RX_SET = 0
GRID_COLUMNS = 181
FIXED_USER_ROW, FIXED_USER_COLUMN = 850, 90
UE_FIRST_ROW, UE_STOP_ROW = 1000, 1200  # 与作者脚本一致，右端不包含。
RIS_SIDE = 32
ACTIVE_RIS_ELEMENTS = 8
TOTAL_SUBCARRIERS = 512
MODEL_SUBCARRIERS = 64
MAX_PATHS = 1  # 论文 Table I。
BANDWIDTH = 100e6
TRANSMIT_POWER_DBW = 5  # 35 dBm。
BATCH_SIZE = 16
SUBCARRIER_CHUNK = 32
VALIDATION_SIZE = 3100
TEST_SIZE = 3100
RANDOM_SEED = 5


def grid_index(row: int, column: int) -> int:
    """MATLAB 的 1 起始行列号转换为 DeepMIMO 的 0 起始用户索引。"""
    return (row - 1) * GRID_COLUMNS + column - 1


def make_channel_parameters() -> dm.ChannelParameters:
    parameters = dm.ChannelParameters()
    parameters.bs_antenna.shape = [RIS_SIDE, RIS_SIDE]
    parameters.bs_antenna.spacing = 0.5
    parameters.ue_antenna.shape = [1, 1]
    parameters.freq_domain = True
    parameters.num_paths = MAX_PATHS
    parameters.ofdm.bandwidth = BANDWIDTH
    parameters.ofdm.subcarriers = TOTAL_SUBCARRIERS
    parameters.ofdm.selected_subcarriers = np.arange(TOTAL_SUBCARRIERS)
    parameters.ofdm.rx_filter = False
    return parameters


def make_ris_codebook(ris_side: int) -> tuple[np.ndarray, np.ndarray]:
    """沿用作者 UPA_codebook_generator.m 的码本与码字顺序。"""
    element = np.arange(ris_side)
    theta_y = np.deg2rad(20.0) - element * np.deg2rad(327.0 / ris_side)
    theta_z = element * np.pi / ris_side
    codebook_y = np.exp(-1j * element[:, None] * theta_y) / np.sqrt(ris_side)
    codebook_z = np.exp(-1j * np.pi * element[:, None] * np.cos(theta_z)) / np.sqrt(ris_side)
    codebook = ris_side * np.kron(codebook_z, codebook_y)
    beam_yz = np.column_stack([np.tile(element, ris_side), np.repeat(element, ris_side)])
    return codebook.astype(np.complex64), beam_yz.astype(np.int16)


def load_paths(user_indices):
    return dm.load(
        SCENARIO, tx_sets=[RIS_BS], rx_sets={UE_RX_SET: user_indices},
        max_paths=MAX_PATHS, compat_v3=True,
    )


def average_relative_rates(cascaded, codebook, channel_scale):
    """先逐子载波 log2(1 + |h.T @ beam / scale|²)，再对全部子载波平均。

    scale 沿用作者训练标签的全局信道幅度缩放；结果不是物理 SNR 下的实际速率。
    分块计算，避免同时保存 [用户, 512, 码字] 的完整中间数组。
    """
    n_users, n_elements, n_subcarriers = cascaded.shape
    rate_sum = np.zeros((n_users, codebook.shape[1]), dtype=np.float64)
    for start in range(0, n_subcarriers, SUBCARRIER_CHUNK):
        channel = cascaded[:, :, start:start + SUBCARRIER_CHUNK].transpose(0, 2, 1)
        response = (channel / channel_scale).reshape(-1, n_elements) @ codebook
        rates = np.log1p(np.abs(response) ** 2) / np.log(2.0)
        rate_sum += rates.reshape(n_users, -1, codebook.shape[1]).sum(axis=1, dtype=np.float64)
    return (rate_sum / n_subcarriers).astype(np.float32)


def make_features(ht, hr, sensors, rng):
    """前 64 个子载波、8 个有源单元；每个复数按 Re, Im 交替排列。"""
    ht_active = ht[sensors, :MODEL_SUBCARRIERS].T.reshape(-1)
    hr_active = hr[:, sensors, :MODEL_SUBCARRIERS].transpose(0, 2, 1).reshape(len(hr), -1)
    # 沿用作者参考脚本的测量噪声约定：Gt=Gr=3 dB，NF=5 dB，处理增益=10 dB。
    noise_dbw = -204 + 10 * np.log10(BANDWIDTH / TOTAL_SUBCARRIERS) + 5 - 10
    noise_power = 10 ** ((noise_dbw - 3 - 3 - TRANSMIT_POWER_DBW) / 10)

    def noise():
        return np.sqrt(noise_power / 2) * (
            rng.standard_normal(hr_active.shape, dtype=np.float32)
            + 1j * rng.standard_normal(hr_active.shape, dtype=np.float32)
        )

    measured = (ht_active + noise()) * (hr_active + noise())
    features = np.stack([measured.real, measured.imag], axis=-1).reshape(len(hr), -1)
    return features, float(np.abs(measured).max())


def generate_dataset(user_indices, output_dir=OUTPUT_PATH,
                     validation_size=VALIDATION_SIZE, test_size=TEST_SIZE):
    user_indices = np.asarray(user_indices, dtype=np.int64)
    n_users = len(user_indices)
    n_train = n_users - validation_size - test_size
    if min(validation_size, test_size) < 0 or n_train <= 0:
        raise ValueError("训练集必须非空，验证集和测试集大小不能为负。")

    dm.config.set("scenarios_folder", str(PROJECT_DIR / "deepmimo_scenarios"))
    parameters = make_channel_parameters()
    ht = load_paths([grid_index(FIXED_USER_ROW, FIXED_USER_COLUMN)]).compute_channels(parameters)[0, 0]
    # 射线路径只读入一次；完整 512 子载波信道按批生成，避免反复读场景文件。
    paths = load_paths(user_indices)
    codebook, beam_yz = make_ris_codebook(RIS_SIDE)
    rng = np.random.default_rng(RANDOM_SEED)
    sensors = np.sort(rng.choice(RIS_SIDE ** 2, ACTIVE_RIS_ELEMENTS, replace=False))
    permutation = np.random.default_rng(RANDOM_SEED).permutation(n_users)
    train, val, test = np.split(permutation, [n_train, n_train + validation_size])
    train_mask = np.zeros(n_users, dtype=bool)
    train_mask[train] = True

    def batches(description):
        for start in tqdm(range(0, n_users, BATCH_SIZE), desc=description, unit="batch"):
            stop = min(start + BATCH_SIZE, n_users)
            hr = paths.trim(idxs=np.arange(start, stop)).compute_channels(parameters)[:, 0].conj()
            yield slice(start, stop), hr

    # 作者相对速率标签需要这个全局尺度；这是计算步骤，不是重复校验。
    channel_scale = 0.0
    for _, hr in batches("1/3 Channel scale"):
        channel_scale = max(channel_scale, float(np.abs(ht * hr).max()))
    channel_scale = max(channel_scale, np.finfo(np.float32).tiny)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_features = 2 * ACTIVE_RIS_ELEMENTS * MODEL_SUBCARRIERS
    x = np.lib.format.open_memmap(output_dir / "X.npy", mode="w+", dtype=np.float32,
                                 shape=(n_users, n_features))
    y = np.lib.format.open_memmap(output_dir / "Y.npy", mode="w+", dtype=np.float32,
                                 shape=(n_users, codebook.shape[1]))
    rate_scale = np.empty(n_users, dtype=np.float32)
    best_ris_index = np.empty(n_users, dtype=np.int32)
    x_scale, train_sum = 0.0, 0.0

    for rows, hr in batches("2/3 X and Y"):
        rates = average_relative_rates(ht * hr, codebook, channel_scale)
        rate_scale[rows] = rates.max(axis=1)
        y[rows] = np.divide(rates, rate_scale[rows, None], out=np.zeros_like(rates),
                            where=rate_scale[rows, None] > 0)
        best_ris_index[rows] = rates.argmax(axis=1)
        features, batch_scale = make_features(ht, hr, sensors, rng)
        x[rows] = features
        x_scale = max(x_scale, batch_scale)
        train_sum += features[train_mask[rows]].sum(dtype=np.float64)

    # 论文 Eq. (10)、(12)：全数据集幅度归一化，再减训练集的一个标量均值。
    x_scale = max(x_scale, np.finfo(np.float32).tiny)
    x_mean = train_sum / (n_train * n_features * x_scale)
    for start in tqdm(range(0, n_users, BATCH_SIZE), desc="3/3 Normalize X", unit="batch"):
        x[start:start + BATCH_SIZE] = x[start:start + BATCH_SIZE] / x_scale - x_mean
    x.flush()
    y.flush()

    arrays = {
        "Ht": ht, "sensor_indices": sensors, "user_indices": user_indices,
        "x_scale": np.float32(x_scale), "x_mean": np.float32(x_mean),
        "full_channel_scale": np.float32(channel_scale), "rate_scale": rate_scale,
        "ris_codebook": codebook, "beam_yz": beam_yz, "best_ris_index": best_ris_index,
        "best_ris_beam_yz": beam_yz[best_ris_index],
        "train_indices": train, "val_indices": val, "test_indices": test,
    }
    for name, array in arrays.items():
        np.save(output_dir / f"{name}.npy", array)
    print(f"X {x.shape}, Y {y.shape}; train/val/test = {len(train)}/{len(val)}/{len(test)}")
    print(f"Saved to {output_dir.resolve()}")


def main():
    users = np.arange(grid_index(UE_FIRST_ROW, 1), grid_index(UE_STOP_ROW, 1))
    generate_dataset(users)


if __name__ == "__main__":
    main()
