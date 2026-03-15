# PID-Net v5 — Training Roadmap

## Phase 1: Foundation ✅ COMPLETE
- Character-level Shakespeare + TinyStories
- Proved architecture works, beats Transformer at all scales
- BPB: 0.228 (TinyStories), 1.63-1.86 (Shakespeare)
- Stagnation-aware D-gate, frequency repetition penalty
- Save/load model weights

## Phase 2: Real Language Model (CURRENT)

### 2.1 Subword Tokenization
Character-level is a toy. Real models need BPE/SentencePiece:
- Use `tiktoken` (GPT-2 tokenizer, 50257 vocab) or train custom BPE
- This changes vocab_size from 65 → 50K+
- Embedding layer becomes the largest parameter cost
- Expect different gate dynamics (tokens carry more info per position)

### 2.2 Serious Datasets
- **OpenWebText** (~38GB) — Wikipedia + web, standard pretraining corpus
- **FineWeb-Edu** (Hugging Face) — filtered web, higher quality
- **SlimPajama** — 627B tokens, deduplicated RedPajama
- Start with a 1GB shard, scale up

### 2.3 Scale to 50-100M Parameters
- d_model=768, n_levels=4, chunk_size=32
- Estimated: ~20-30M params (PID-Net) vs ~100M (Transformer baseline)
- Need Mac Studio (64GB+ RAM) or cloud GPU for this scale
- Multiple random seeds for error bars

### 2.4 Training Infrastructure
- Gradient accumulation (effective batch size 32-64)
- Mixed precision (float16 where possible)
- Learning rate warmup + cosine decay
- Proper train/val split with periodic eval
- Checkpoint saving every N steps
- WandB or MLX equivalent for experiment tracking
- Proper BPB on held-out validation set (not training set)

### 2.5 Evaluation
- **Perplexity on WikiText-103** — standard benchmark
- **LAMBADA** — long-range prediction
- **HellaSwag** — commonsense reasoning (via few-shot)
- **BPB on held-out validation** — honest metric
- Compare against published Transformer numbers at same param count

## Phase 3: Architecture Refinements
- Scheduled sampling (mix teacher-forced + autoregressive)
- Adaptive rewrite steps (more steps for hard tokens)
- Edge evolution every 2nd step (speed optimization)
- MLX Metal kernel fusion for graph ops
- Long-context evaluation (4K+ tokens via fractal hierarchy)

## Phase 4: Real Applications
- Fine-tune on instruction data (Alpaca, ShareGPT)
- Chat/QA evaluation
- Code generation
- Integration as Ion's actual cognitive backend

## Hardware Path
- M2 MacBook Air 8GB: up to ~10M params ← current
- Mac Studio M2 Ultra 192GB: up to ~1B params ← next purchase
- Mac Studio cluster: multi-billion scale
