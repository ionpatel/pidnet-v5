import mlx.core as mx
from pidnet.fractal import FractalPIDNet

model = FractalPIDNet(65, 384, 272, 3, 8, 16, 3)
model.load_weights('model_5k.safetensors')

tokens = mx.array([[23, 21, 26, 19]])  # "KING"
logits, diag = model(tokens)
last = logits[0, -1, :]
mx.eval(last)

print("Last token logits (first 10):")
for i in range(10):
    print(f"  [{i}] = {last[i].item():.6f}")

print(f"  max={mx.max(last).item():.4f} min={mx.min(last).item():.4f}")
print(f"  argmax={mx.argmax(last).item()}")
print(f"Gate P={diag['gate_p'].item():.4f} I={diag['gate_i'].item():.4f} D={diag['gate_d'].item():.4f}")
