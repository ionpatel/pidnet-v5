# Apple Silicon Deep Dive: M2 MacBook Air for PID-Net

## Hardware Architecture

### M2 Chip Layout
```
┌─────────────────────────────────────────────────────────┐
│                     M2 SoC                               │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ CPU      │  │ GPU      │  │ Neural Engine         │   │
│  │ 8 cores  │  │ 8-10     │  │ 16 cores              │   │
│  │          │  │ cores    │  │ 15.8 TOPS (INT8)      │   │
│  │ 4P + 4E  │  │ 128 EUs  │  │ Matrix multiply       │   │
│  │          │  │          │  │ Convolution            │   │
│  └────┬─────┘  └────┬─────┘  └──────────┬────────────┘   │
│       │              │                    │               │
│  ┌────┴──────────────┴────────────────────┴────────────┐  │
│  │            UNIFIED MEMORY CONTROLLER                 │  │
│  │          100 GB/s bandwidth (LPDDR5)                 │  │
│  │         ALL compute units share this pool            │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │                                 │
│  ┌──────────────────────┴──────────────────────────────┐  │
│  │              8GB UNIFIED MEMORY                      │  │
│  │     No CPU↔GPU copy! Zero-copy shared access.       │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │    SSD Controller: ~1.5-2.5 GB/s (256GB NAND)      │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Specs: M2 MacBook Air (8GB)

| Component | Spec | PID-Net Relevance |
|-----------|------|-------------------|
| **CPU** | 4 Performance + 4 Efficiency cores | Tokenization, control flow, beam search |
| **GPU** | 8 cores, 128 execution units | Matmuls, message passing, edge evolution |
| **Neural Engine** | 16 cores, 15.8 TOPS (INT8) | Embedding lookup, gate softmax, activation functions |
| **Unified Memory** | 8GB LPDDR5, 100 GB/s bandwidth | Active model weights + graph states |
| **SSD** | 256GB, ~1.5-2.5 GB/s seq read | Cold graph states, weight offloading |
| **Memory Bus** | 128-bit, 100 GB/s | Single bottleneck for all compute |
| **Process** | 5nm, 20B transistors | Power efficiency for 24/7 operation |
| **TDP** | ~15-22W under load | Can run inference indefinitely without thermal throttle |

## How PID-Net Maps to M2

### Current State (What We Use)

```
GPU only ──────────────────────────── ~60% utilized
CPU ───────────────────────────────── Python overhead only
Neural Engine ─────────────────────── 0% (UNUSED!)
Memory ────────────────────────────── ~50-70% (model + training data)
SSD ───────────────────────────────── Data loading only
```

### Optimal State (What We Should Use)

```
GPU ───── Message passing matmuls ──── Dense matrix ops (W_msg @ nodes)
         Edge evolution ────────────── Dot products for affinity
         Fractal pooling/broadcast ─── Weighted sums across levels
         Fast weight outer products ── I-stream writes (v ⊗ k)

CPU ───── Tokenization ────────────── Sequential text → token IDs
         Beam search / sampling ────── Token selection logic
         Weight paging scheduler ───── SSD↔RAM management
         Graph state serialization ─── Save/load cold levels
         Energy conservation ───────── Norm computation + clipping

Neural ── Embedding lookup ─────────── Table lookup (optimized fixed-func)
Engine    Gate softmax ─────────────── softmax(W_g @ [P,I,D,X] / T)
          LayerNorm ────────────────── Normalization (ANE-optimized)
          Sigmoid/GELU activations ─── Fixed-function nonlinearities

Memory ── PID weights (shared!) ────── One rule set, used at ALL levels
(RAM)     Level 0 graph state ──────── Token-level nodes + edges
          Level 1 graph state ──────── Chunk-level nodes + edges
          Fast weight matrix W ─────── I-stream associative memory

SSD ───── Level 2+ graph states ────── Cold fractal levels
          Checkpoints ──────────────── Model snapshots
          Token cache ──────────────── Pre-tokenized datasets
          Prefetch buffer ──────────── D-stream predicted states
```

## Deep Analysis: Each Component

### 1. GPU (8 cores, 128 EUs)

**What it does in PID-Net:**
- `W_msg @ nodes` — message passing: (d×d) @ (B×N×d) 
- `nodes @ nodes.T` — edge evolution affinity: (B×N×d) @ (B×d×N)
- `v_wr ⊗ k_wr` — fast weight outer product: (B×d×1) @ (B×1×d)
- `W_fast @ x` — fast weight read: (B×d×d) @ (B×N×d)
- `W_pred @ x` — D-stream prediction: MLP forward pass

**Bottleneck analysis:**
At d=384, the key matmul is 384×384 = 147K multiply-adds per node.
With N=256 nodes, batch=4: 256 × 4 × 147K = 150M multiply-adds per rewriting step.
3 rewriting steps × 3 fractal levels = 9 passes = 1.35B multiply-adds per forward pass.

M2 GPU: ~3.6 TFLOPS (FP32) → theoretical: 1.35B / 3.6T = 0.37ms per forward pass.
Actual: ~5ms (Python overhead, memory access patterns, small batch inefficiency).

**Optimization opportunities:**
- Metal kernel fusion: combine message passing + norm + gate into one kernel
- Use FP16 (M2 GPU does 7.2 TFLOPS in FP16 — 2x speedup)
- Batch graph ops: process all fractal levels in one fused kernel

### 2. CPU (4P + 4E cores)

**What it does in PID-Net:**
- Python interpreter overhead (dominant cost at small model sizes!)
- Tokenization: SentencePiece encode/decode
- Sampling: argmax / categorical / top-k filtering
- Control flow: loop management, conditional branching
- Memory management: garbage collection, array allocation

**Bottleneck analysis:**
At 25 tok/s inference, CPU overhead is ~42% of total time.
The Python loop calling model() per token is the killer.
Each call has: Python→MLX dispatch, array creation, result extraction.

**Optimization opportunities:**
- Compile the generation loop with `mx.compile()` (MLX's JIT)
- Move sampling to GPU (top-k + categorical in one kernel)
- Batch multiple tokens in speculative decoding
- Cython/C extension for the hot loop

### 3. Neural Engine (16 cores, 15.8 TOPS)

**What it does currently:** NOTHING. Completely idle.

**What it COULD do:**
The Apple Neural Engine (ANE) is optimized for:
- Matrix multiplications on fixed shapes
- Convolutions
- Element-wise operations
- Activation functions (ReLU, sigmoid, GELU, softmax)

**PID-Net operations perfect for ANE:**
1. **Embedding lookup**: Table lookup is ANE-native
2. **Gate softmax**: softmax(W @ input / T) — classic ANE op
3. **LayerNorm**: Normalization is ANE-optimized
4. **Activation functions**: sigmoid (stagnation gate), GELU (MLP)
5. **Energy normalization**: Norm compute + scale

**How to use ANE with MLX:**
MLX doesn't directly target ANE (it targets GPU + CPU).
Options:
- CoreML: Convert model to CoreML format, ANE auto-selected for compatible ops
- Custom Metal + ANE dispatch: Write Metal shaders that hint ANE usage
- Wait for MLX ANE support (Apple is working on it)

**Estimated impact:** 
If we offload embedding + gate + norms to ANE, those ops become essentially free
(ANE runs in parallel with GPU). This could give 15-25% inference speedup.

### 4. Unified Memory (8GB LPDDR5, 100 GB/s)

**What it does in PID-Net:**
- Stores model weights (~3.4M params × 4 bytes = 13.6MB)
- Stores graph states (nodes + edges + fast weights)
- Stores training data in memory
- Python interpreter + OS overhead (~2-3GB)

**Memory budget at d=384, N=256, B=4:**
```
Model weights:        13.6 MB  (3.4M params, float32)
Embedding (1K vocab):  1.0 MB  (1024 × 256 × 4)
Graph nodes:           1.0 MB  (4 × 256 × 256 × 4)
Graph edges:           1.0 MB  (4 × 256 × 256 × 4)
Fast weights:          1.5 MB  (4 × 256 × 256 × 4)  
Predictions:           1.0 MB  (4 × 256 × 256 × 4)
Gradients (training):  13.6 MB (same as weights)
Optimizer states:      27.2 MB (Adam: 2 × weights)
─────────────────────────────
Model total:           ~60 MB
Training data:         ~50 MB (OpenWebText sample)
Python + OS:           ~2.5 GB
─────────────────────────────
TOTAL:                 ~2.6 GB of 8 GB used
FREE:                  ~5.4 GB
```

**At 50M params (Mac Studio target):**
```
Model weights:         200 MB
Gradients:             200 MB
Optimizer:             400 MB
Graph states:          ~50 MB (larger N)
─────────────────────────────
TOTAL:                 ~1 GB + Python + data
```
→ 50M params fits comfortably in 8GB for inference, needs 64GB+ for training.

**The unified memory advantage:**
On discrete GPU systems (NVIDIA), weights must be copied CPU→GPU via PCIe:
- PCIe 4.0: 32 GB/s (3x slower than M2 memory bus)
- PCIe 5.0: 64 GB/s (still slower)
- Each forward pass on discrete GPU: potential PCIe bottleneck

On Apple Silicon: weights are ALREADY where GPU needs them. Zero copy.
This is why Apple Silicon excels at inference for models near the memory limit.

### 5. SSD (256GB, ~1.5-2.5 GB/s)

**What it does in PID-Net:**
- Stores training data files
- Checkpoint saving
- (Future) Cold graph state storage

**For tiered inference:**
At 2 GB/s sequential read, loading one fractal level's graph state:
- Level 2 state (~1MB): 0.5ms — imperceptible
- Level 3 state (~10MB): 5ms — one token generation time
- Large checkpoint (~200MB): 100ms — noticeable pause

**D-stream prefetch latency budget:**
If we prefetch one step ahead, we have ~40ms (at 25 tok/s) to load from SSD.
At 2 GB/s, that's 80MB of prefetch capacity per token — more than enough.

**SSD endurance concern:**
Constant SSD paging for inference would write ~10GB/hour.
At 150 TBW (typical 256GB SSD rating): 15,000 hours = 1.7 years continuous.
→ Fine for development, but production Mac Studios should use larger SSDs.

## PID-Net Operations Breakdown

### Forward Pass Compute Map

```
Token Input
    │
    ├─[ANE] Embedding lookup ────────────── 0.01ms
    ├─[ANE] Positional embedding ─────────── 0.01ms
    │
    ▼
For each rewriting step (×3):
    │
    ├─[GPU] P-stream: message passing ────── 0.5ms
    │   └── W_msg @ nodes, adjacency mul
    │
    ├─[GPU] I-stream: fast weight read ───── 0.3ms  
    │   └── W_fast @ nodes
    │   └── [GPU] fast weight write (D-gated outer product)
    │
    ├─[GPU] D-stream: prediction error ───── 0.4ms
    │   └── actual - predicted
    │   └── MLP(error)
    │   └── [GPU] new prediction = MLP(nodes)
    │
    ├─[ANE] Gate computation ─────────────── 0.02ms
    │   └── softmax(W_g @ [P,I,D,X] / T)
    │
    ├─[GPU] Weighted combination ─────────── 0.1ms
    │   └── g_P·P + g_I·I + g_D·D
    │
    ├─[CPU] Energy conservation ──────────── 0.05ms
    │   └── norm ratio + clip
    │
    └─[GPU] Edge evolution ───────────────── 0.6ms
        └── affinity scores + sigmoid + topk
    
    Total per step: ~2.0ms
    × 3 steps = 6.0ms
    
For each fractal level (×3):
    │
    ├─[GPU] Pooling (bottom-up) ──────────── 0.2ms
    ├─[GPU] Rewriting (same weights!) ────── 2.0ms
    └─[GPU] Broadcasting (top-down) ──────── 0.2ms
    
    Total fractal: ~7.2ms

[ANE] Readout norm + logits ──────────────── 0.05ms
[CPU] Sampling + penalty ─────────────────── 0.3ms
─────────────────────────────────────────────────
TOTAL per token: ~13.5ms theoretical
ACTUAL: ~40ms (Python overhead dominant)
Optimization potential: 3x with kernel fusion + compiled loop
```

## Optimization Roadmap

### Quick Wins (Days)
1. **`mx.compile()`** the forward pass — JIT compilation eliminates Python dispatch overhead
2. **FP16 inference** — 2x faster matmuls, half memory usage
3. **Fuse P+I+D gate** into single kernel — reduce memory round-trips

### Medium Effort (Weeks)  
4. **CoreML export** for ANE utilization — embedding + gate + norms on Neural Engine
5. **Speculative decoding** — draft with small model, verify with full model
6. **KV-cache equivalent** — cache graph states between tokens (avoid recompute)

### Hard Mode (Months)
7. **Custom Metal kernels** — fused message passing + edge evolution
8. **ANE direct dispatch** — custom ANE programs for PID-specific ops
9. **Multi-Mac distributed inference** — split fractal levels across machines
10. **Quantization** — INT8/INT4 weights with calibration (ANE loves INT8)

## Comparison: M2 Air vs M2 Ultra vs M5 Ultra (Projected)

| Spec | M2 Air | M2 Ultra | M5 Ultra (est.) |
|------|--------|----------|-----------------|
| CPU cores | 8 | 24 | 32+ |
| GPU cores | 8 | 76 | 128+ |
| Neural Engine | 16-core | 32-core | 48+ core |
| Memory | 8GB | 192GB | 512GB-1TB |
| Memory BW | 100 GB/s | 800 GB/s | 1.2+ TB/s |
| SSD speed | 2.5 GB/s | 7.4 GB/s | 14+ GB/s |
| Max model (FP16) | ~3M PID | ~90M PID | ~250B-500B PID |
| Est. tok/s (10B) | N/A | ~30 | ~100+ |
| Est. tok/s (100B) | N/A | ~3 | ~15+ |
| Power | 15W | 200W | 250W |
| Price | $1,200 | $8,000 | $12,000+ |
