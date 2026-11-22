# RIS 数据集生成

参考论文 *On-Device Deep Learning for RIS Beamforming: Design and Latency-Rate Trade-Offs*
（本地 `D:/RIS/STM32_Deep_Learning.pdf`，Table I、Eq. (3)、(6)、(10)–(12)）及
[作者参考实现](https://github.com/LucaUrbinati44/DL-on-RISC/tree/main/code)。

## 数据口径

- 场景 `o1_28`，RIS 为 32×32，固定随机选择 8 个有源单元，每条链路取 1 条路径。
- **X：前 64 个子载波**，每个样本 8×64 个带噪级联复信道，拆成 1024 个实数。
  子载波在外层、单元在内层，每个复数按 `Re, Im` 交替排列；沿用作者实际 MATLAB
  `reshape([real(H_bar) imag(H_bar)].',[],1)` 的顺序。
  先除以整个数据集带噪复信道的最大幅度，再减去**仅由训练集计算的一个标量均值**。
  `X.npy` 已完成这两步，训练脚本不要再做一次中心化。
- **Y：全部 512 个子载波**参与每个码字的评分。必须先逐子载波计算 `log2(1 + 功率)`，
  再平均；不能先平均信道/功率再取对数，也不能先对每个子载波的标签归一化再平均。
  最后以每个用户的最大评分归一化，得到 1024 维回归标签，最大元素对应最佳码字。
- X、Y 的第 i 行对应同一个 `user_indices[i]`。不单独筛选或打乱 X、Y；划分索引同时作用于两者。
  完整数据共 36,200 个用户，训练/验证/测试为 **30,000 / 3,100 / 3,100**。

## 速率缩放与论文公式的区别

本次修正了子载波平均范围，**保留原项目/作者训练脚本的相对速率缩放**：

```text
C[u,m,k] = Ht[m,k] * conj(Hr_raw[u,m,k])
D = max(abs(C))                         # 全用户、全单元、512 个子载波
r[u,b] = mean_k(log2(1 + abs(sum_m(C[u,m,k] * codebook[m,b]) / D)**2))
Y[u,b] = r[u,b] / max_b(r[u,b])
```

论文 Eq. (3) 使用物理 SNR，而作者公开训练脚本采用上述 D 缩放，且只算第一个子载波。
这里按用户要求把后者改为完整 512 个子载波的平均；因此不能声称整个仿真已严格复现论文物理模型。
`Y * rate_scale[:, None]` 还原的是归一化前的**相对速率评分**，不是实际 bit/s/Hz。
要报告“达到理论最优速率的百分之几”，还需用同一物理 SNR 计算预测码字和最优码字的实际速率，
不能用预测分数的最大值代替实际速率。

X 的测量噪声沿用作者脚本：发射功率改为 5 dBW（35 dBm），Gt=Gr=3 dB、NF=5 dB、
处理增益 10 dB；天线模型及码本沿用原项目，不视为与论文所有物理参数逐项等价。

## 运行

在项目目录使用现有虚拟环境：

```powershell
cd D:\RIS\deepmimo
.\.venv\Scripts\python.exe test.py
.\.venv\Scripts\python.exe generate_full_dataset.py
```

第一条 Python 命令生成 100 个调试样本到 `stm32_sample_512`；第二条生成完整数据到
`stm32_dataset_512`。新数据需要重新生成，旧 `stm32_dataset` 和 `stm32_dataset.zip` 不会自动更新。
在 Linux 上也使用文件名 **`X.npy`、`Y.npy`**，注意大小写。

训练读取 `train_indices.npy`、`val_indices.npy`、`test_indices.npy`。
部署预处理使用同一套 `sensor_indices.npy`、`x_scale.npy`、`x_mean.npy`；
`best_ris_index.npy` 保存每个样本在 512 子载波平均评分下的最佳码字。

只加载一次用户射线路径，分批生成信道、分块计算码字评分，以内存映射写入数据。
`BATCH_SIZE` 和 `SUBCARRIER_CHUNK` 可调小以降低内存占用，不改变子载波平均范围。
全局幅度扫描、特征生成、最终归一化是必要的计算步骤；重复的形状、有限值、目录检查已移除。
