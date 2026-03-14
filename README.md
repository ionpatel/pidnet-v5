# PID-Net v5: Hypergraph Rewriting Cognitive Architecture

**Intelligence as iterated hypergraph rewriting — not function approximation.**

## What is this?

A neural architecture where cognition emerges from applying learned PID (Proportional-Integral-Derivative) rewriting rules on a dynamic hypergraph. Inspired by Wolfram Physics, PID Control Theory, and Predictive Coding.

- **P-stream (Perceive):** Message passing over the current graph state
- **I-stream (Integrate):** Fast weight associative memory for past states
- **D-stream (Differentiate):** Prediction error — what surprised the model
- **Gate:** Selects which rewriting rule to apply

## Why?

Every existing architecture (Transformer, Mamba, RWKV, Griffin) uses a **fixed computation graph**. Intelligence is approximated by learning weights within that static structure.

PID-Net v5 has a **dynamic computation graph** that evolves during processing. Intelligence *emerges* from iterated rewriting — like how spacetime emerges from simple rules in Wolfram Physics.

## Framework

**MLX** (Apple Silicon native) — not PyTorch.

Why: Our architecture is sparse graph operations with irregular memory access. Apple Silicon's unified memory eliminates the CPU↔GPU transfer bottleneck that kills graph ops on CUDA. The architecture and hardware are aligned.

**Scaling path:** MacBook → Mac Studio → Mac Studio Cluster

## Structure

```
pidnet-v5/
├── pidnet/              # Core library
│   ├── core/            # Graph primitives, state management
│   ├── streams/         # P, I, D stream implementations
│   ├── gate/            # Gating & rule selection
│   ├── graph/           # Dynamic topology, edge evolution
│   ├── memory/          # Fast weight programmer, multi-timescale
│   └── model.py         # Full PIDGraphNet model
├── tests/               # Unit & integration tests
│   ├── unit/            # Component tests
│   └── integration/     # End-to-end tests
├── train/               # Training scripts
│   ├── train.py         # Main training loop
│   ├── data.py          # Data loading & tokenization
│   └── diagnostics.py   # Real-time dashboard
├── experiments/         # Experiment configs & results
├── benchmarks/          # Performance benchmarks
└── scripts/             # Utility scripts
```

## Branch Strategy

- `main` — stable, tested releases only
- `develop` — integration branch (PRs merge here first)
- `phase-1/foundation` — Fixed graph + PID rewriting
- `phase-2/dynamic-topology` — Edge evolution, node eviction
- `phase-3/predictive-coding` — D-stream prediction error, D→I coupling
- `phase-4/fractal` — Multi-scale, shared rules, adaptive chunking
- `phase-5/scale` — Scale to 125M+, benchmarks

## Research

Architecture spec, research papers, and theoretical foundations live in the **research repo**: [ionpatel/pid-net](https://github.com/ionpatel/pid-net)

## Status

🚧 **Phase 1: Foundation** — Building core graph primitives and PID rewriting step

## Authors

Ion Patel & Harshil Patel
