# OSCLLM Phase 1a - 流程验证版

## 📦 已创建的文件

| 文件 | 说明 |
|------|------|
| `configs/oscllm_1b_phase1a.json` | 模型配置 (D=384, L=16, vocab=512) |
| `train/train_deepspeed.py` | DeepSpeed 训练入口 |
| `src/oscllm/training/engine.py` | 训练引擎 (ZeRO-2 + BF16) |
| `run.sh` | 启动脚本 (NCCL/Numa/OMP绑定) |

## 📊 模型配置

| 参数 | 值 |
|------|-----|
| vocab_size | 512 (简化版 BPE tokenizer) |
| d_model | 384 |
| num_layers | 16 |
| seq_len | 4096 |
| params | ~7.7M |

## 🗃️ 数据资源

| 类型 | 目录 | 规模 |
|------|------|------|
| 中文通用文本 | data/chinese_c4/ | ~30KB (合成样本) |
| 唐诗+论语 | data/poetry_ancient/ | ~150KB (样本) |
| 训练 corpus | dataset/corpus.bin | 68K tokens |

## 🚀 启动训练

```bash
cd /home/dja/桌面/OSC-LLM

# 环境变量
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# 启动 (双卡 BF16)
numactl --cpunodebind=0 --membind=0 \
deepspeed --num_gpus 2 train/train_deepspeed.py \
  --config configs/oscllm_1b_phase1a.json \
  --bin dataset/corpus.bin
```

## 📈 预期结果

| 指标 | 值 |
|------|-----|
| 参数量 | ~7.7M |
| Token/step | 1,048,576 (batch_size=4 × ga=4 × seq_len) |
| 预估时长 | ~3-5 小时 (双卡 BF16) |

## 🔧 技术细节

- **优化器**: AdamW (bf16)
- **学习率**: 3e-4 (warmup 1k steps + cosine decay)
- **梯度裁剪**: max_norm=1.0
- **分布式**: DeepSpeed ZeRO-2 + BF16

## ⚠️ 注意事项

当前 vocab_size=512 是简化测试版本。正式训练需替换为 Qwen/Qwen2.5-0.5B 的 151K 词表。

---

**项目路径**: `/home/dja/桌面/Test Train OSC-LLM`
