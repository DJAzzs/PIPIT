# PIPIT（原名 OSC-LLM）— 频域动态深度大语言模型

> **P 系 · 系统/联合创新**
> 把「复频域分解 + 动态深度 + 稀疏脉冲路由」三条正交路线融合进一个可训练的 LLM 骨架。
> 核心主张：**LLM 不必每层都用满算力** —— 把有限算力花在真正需要的 token 与位置上，
> 同时把注意力复杂度从 O(n²) 降到 O(n log n)。

## 设计动机

LLM 沿序列/沿层的计算是均匀的，但真实文本的信息量并不均匀：关键位置需要深层推理，
大多数常规 token 只需要浅处理。PIPIT 用一个骨架同时探索三件事：

| 机制 | 全称 | 作用 |
|---|---|---|
| **CFRA** | 复频域谐振注意力 (Complex-Frequency Resonant Attention) | 在频域做全局信息压缩，O(n log n) |
| **LTK** | 液体时变核 (Liquid Time-Kernel) | 沿时间轴自适应计算深度（RK4/ODE） |
| **SFR** | 脉冲发放路由 (Spiking-Firing Router) | 条件计算 / 稀疏激活 |

## 📦 训练配置

> **状态声明（如实）**：本项目当前仅完成 **84M 参数的快速验证**（架构 / 训练流程走通）。
> **1B 为"配置目标"**，配置已定义但**尚未在足量数据上完成训练**——训练数据目前仅为
> 极小规模合成样本（见下文），**未达到可训练 1B 模型所需的数据量**。请勿将 1B 视为
> 已获得的成果。

| 规模 | 模型参数 | 配置文件 | 状态 |
|------|---------|---------|------|
| **1B（配置目标）** | 1.0B | `configs/oscllm_1b.json` | 配置已定义，**未完成训练** |
| 快速验证 | 84M | `configs/oscllm_10m.json` | ✅ 已完成验证 |

### 1B 配置（目标定义，非已训练成果）
```json
{
  "d_model": 1024,
  "n_layer": 80, 
  "vocab_size": 151936 (Qwen-0.5B tokenizer)
}
```

## 🚀 启动训练

```bash
cd OSC-LLM  # 或你的仓库本地路径

# 环境变量
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=8 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0

# 启动 (双卡 BF16)
numactl --cpunodebind=0 --membind=0 \
deepspeed --num_gpus 2 train/train_deepspeed.py \
  --config configs/oscllm_1b.json \
  --bin dataset/corpus.bin
```

## 📊 配置文件说明

| 文件 | 参数量 | 主要用途 |
|------|--------|---------|
| `configs/oscllm_1b.json` | ~1.0B | 标准预训练 (官方目标) |
| `configs/oscllm_10m.json` | ~84M | 快速流程验证 |

## 📁 数据现状（如实）

> **当前数据仅用于验证训练管线，不构成真实训练集。** 仓库内的
> `data/chinese_c4/shard_*.txt` 为**合成的重复占位样本**（仅 5 句话重复填充，
> 用于冒烟测试 pipeline），**不是真实语料**。完整语料的下载与预处理脚本见
> `scripts/download_data.py`，需扩展至足量 token 才能进行正式训练。

- **中文通用**: `data/chinese_c4/*.txt` — ⚠️ 重复占位样本（冒烟用）
- **诗词/古籍**: `data/poetry_ancient/*.txt` — ~150KB 真实古籍样本（少量示例）
- **正式训练语料**: 需扩展为 1B+ tokens（建议 ModelScope ChineseC4 或 shuge.org）

### ⚠️ 诚实说明：尚未完成正式训练

PIPIT 当前成果**仅限于 84M 模型在小规模数据上的流程验证**——验证了
CFRA/LTK/SFR 架构与 DeepSpeed 训练链路可运行并收敛。**没有**完成 1B 模型的
有效训练，也**没有**任何质量评测结果。这是一个实验性架构项目而非已训练好的模型。

## ⚙️ 技术特点

- **CFRA**: 复频域谐振注意力 - O(n log n) 复杂度
- **LTK**: 液体时变核 - 自适应计算深度 (RK4)
- **SFR**: 脉冲发放路由 - 条件计算/稀疏激活

## 🔧 系统要求

- GPU: 2× RTX PRO 6000 Max-Q (192GB total VRAM)  
- CUDA: 13.0+ (支持 FP8)
- 深度学习框架: PyTorch 2.13+, DeepSpeed

## 路线图

- [x] CFRA / LTK / SFR 三机制实现，84M 快速验证
- [x] 1B 配置与 DeepSpeed 双卡训练脚本
- [ ] 扩展语料至 1B tokens（ChineseC4 / shuge 古籍）
- [ ] 与同规模标准 Transformer 的等算力对照实验
- [ ] 动态深度 / 稀疏激活的 token 级诊断可视化

## License

Apache License 2.0 · Copyright (c) 2025 DJAzzs
