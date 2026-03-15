# PID-Net C++ Inference Engine

Native C++ inference engine for PID-Net on Apple Silicon.
Uses MLX C++ API directly — no Python, no GIL, no garbage collector.

## Target Performance
- Current (Python + mx.compile): 104 tok/s
- Target (C++): 500+ tok/s
- Ultimate (C++ + Metal fusion): 1000+ tok/s

## Architecture
```
pidnet_engine.cpp    — Main inference loop + CLI
pidnet_model.h       — Model definition using MLX C++ API  
pidnet_generate.h    — Token generation with repetition penalty
pidnet_tokenizer.h   — SentencePiece integration
CMakeLists.txt       — Build system
```

## Build
```bash
mkdir build && cd build
cmake .. -DMLX_DIR=/path/to/mlx
make -j$(nproc)
```

## Usage
```bash
./pidnet generate --model weights.safetensors --prompt "The United States"
./pidnet benchmark --model weights.safetensors --tokens 100
```
