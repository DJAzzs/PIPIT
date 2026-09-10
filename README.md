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

| 规模 | 模型参数 | 配置文件 |
|------|---------|---------|
| **1B (官方目标)** | 1.0B | `configs/oscllm_1b.json` |
| 快速验证 | 84M | `configs/oscllm_10m.json` |

### 1B 模型配置详情
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

## 📁 数据准备

当前使用的数据：
- **中文通用**: `data/chinese_c4/*.txt` (~30KB, 合成样本)
- **诗词/古籍**: `data/poetry_ancient/*.txt` (~150KB, 样本)

正式训练需扩展为 1B tokens (建议从 ModelScope 下载 ChineseC4 或 shuge.org 爬虫)。

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

MIT License · Copyright (c) 2025 DJAzzs
