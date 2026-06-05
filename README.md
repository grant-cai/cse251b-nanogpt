# Parameter-Efficient GPT — CSE 251B NanoGPT Competition

A GPT-style language model trained **from scratch** under a strict **≤100M-parameter**
budget for the CSE 251B NanoGPT Competition. The goal: lowest perplexity on a held-out,
multi-domain test set, on a limited single-GPU compute budget.

## Results

| Metric | Value |
|---|---|
| **Validation perplexity** | **29.20** |
| Parameters | 95.27M (under the 100M cap) |
| Tokenizer | GPT-2 BPE (vocab 50,257) |
| Context length | 1,024 tokens |
| Training hardware | Single NVIDIA RTX 4070 (12 GB) |
| Tokens processed | ~2.5B (past Chinchilla-optimal for the model size) |

Perplexity improved over the course of the project from an initial **31.0 → 30.6 → 29.2**
through extended training and a learning-rate cooldown (see *Training*).

## Architecture

Built on the nanoGPT framework, with parameter-efficiency techniques chosen to maximize
quality per parameter within the 100M cap:

- **Grouped-Query Attention (GQA)** — `n_head=12`, `n_kv_heads=4`, reducing key/value
  projection parameters and KV-cache cost relative to full multi-head attention.
- **ALiBi positional encoding** — attention-bias positions instead of learned positional
  embeddings, saving parameters and improving length generalization.
- **SwiGLU** feed-forward activations.
- **Weight tying** between the token embedding and the output (LM head) projection.
- **9 transformer layers** (`n_layer=9`), tuned to fit the parameter budget.

## Training

### Dual-optimizer setup
Two disjoint optimizers update disjoint parameter sets in the same step:

- **Muon** (Newton–Schulz orthogonalization) on the 2D weight matrices
  (attention/MLP projections).
- **AdamW** on embeddings, the LM head, and all 1D parameters (norms, biases),
  where orthogonalization does not apply.

### Configuration
- **Effective batch size:** 64 sequences (= micro-batch × gradient-accumulation;
  e.g. 8 × 8 or 16 × 4), ≈ 65,536 tokens per optimizer step.
- **Mixed precision:** bf16 autocast with fp32 master weights.
- **`torch.compile`** for ~8× throughput vs. eager execution.
- **Learning-rate schedule:** cosine decay with warmup, followed by a **warmup-stable-decay
  (WSD) cooldown** — annealing a converged checkpoint down to a low LR, which delivered
  the final perplexity gain.

### Data
- **FineWeb-Edu** (10B-token sample) and **CommonPile**, mixed and rotated across shards
  during training. Data loaded via memory-mapped `.npy` shards.

## Engineering / systems work

Training was run on a single 12 GB consumer GPU, which required diagnosing and resolving
a number of real-world constraints:

- **GPU under-utilization:** identified silent `torch.compile` fallback to eager mode (the
  tell: high GPU-util but low power draw) and restored it by freeing VRAM headroom via
  micro-batch/grad-accum rebalancing — **8 s → 1 s per iteration**.
- **CUDA OOM / memory fragmentation:** mitigated with `expandable_segments`, batch-size
  tuning, and memory-light kernels (replacing full-vocab sorts with top-k reductions).
- **Host-RAM exhaustion:** eliminated WSL crashes caused by loading multi-GB teacher-logit
  arrays into RAM, by switching to memory-mapped (`mmap_mode='r'`) access.
- **Checkpointing:** debugged best-val-only save logic (`always_save_checkpoint`) that was
  silently discarding progress on resume.

## Knowledge distillation (explored)

Prototyped **offline knowledge distillation** from a larger teacher (GPT-2 medium) as an
alternative training path:

- Precomputed sparse **top-p-k** teacher distributions (top_k=100, top_p=0.95) to disk to
  avoid co-resident teacher+student memory.
- Forward-KL loss on the sparse targets, blended with cross-entropy on a WSD weight
  schedule.

The teacher measured ~20 ppl on the validation set, but pure-KD distillation regressed the
already-converged student (matching the teacher's truncated distribution traded away
ground-truth calibration). The from-scratch + cooldown path was retained as the final
submission. The distillation experiment is preserved as `distillation.ipynb`.

## Reproduce

Train (continued-training / cooldown example):

```bash
python train.py --init_from=resume --out_dir=out \
  --batch_size=8 --gradient_accumulation_steps=8 \
  --warmup_iters=<resume_iter> --learning_rate=2e-4 --min_lr=1e-6 \
  --max_iters=<resume_iter+N> --lr_decay_iters=<resume_iter+N> \
  --eval_interval=250 --eval_iters=100 \
  --always_save_checkpoint=False --compile=True --wandb_log=False
```

Evaluate (same harness the graders use):

```bash
# local checkpoint
python evaluate.py --model_dir my_submission --data val.bin

# verify the uploaded HuggingFace submission
python evaluate.py --hf_repo <user>/<repo> --data val.bin
```

The submission directory must contain `model.py` (defining
`load_model(checkpoint_path, device)`) and `checkpoint.pt`.

## Key takeaways

- A carefully parameter-budgeted from-scratch model (GQA + ALiBi + SwiGLU + weight tying)
  reached **29.2 ppl** within a 100M cap on a single consumer GPU.
- Past Chinchilla-optimal, a fixed-size model keeps improving with more training — and a
  late-stage LR cooldown is a cheap, reliable way to extract that last gain.
- On constrained hardware, throughput and memory engineering (compile, mmap, batch
  rebalancing) mattered as much as modeling choices.
