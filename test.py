"""用同一套生成逻辑生成 100 个调试样本，不覆盖完整数据集。"""

import numpy as np

from generate_full_dataset import PROJECT_DIR, UE_FIRST_ROW, generate_dataset, grid_index


if __name__ == "__main__":
    first_user = grid_index(UE_FIRST_ROW, 1)
    generate_dataset(np.arange(first_user, first_user + 100),
                     PROJECT_DIR / "stm32_sample_512", validation_size=10, test_size=10)
