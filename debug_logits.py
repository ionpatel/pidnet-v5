import mlx.core as mx
from pidnet.fractal import FractalPIDNet

model = FractalPIDNet(65, 384, 272, 3, 8, 16, 3)
model.load_weights('model_5k.safetensors')

tokens = mx.array([[23, 21, 26, 19]])  # "KING"

# Step 1: Check embedding
positions = mx.arange(4)
tok_embed = model.embed(tokens)
pos_embed = model.pos_embed(positions)
nodes = tok_embed + pos_embed
mx.eval(nodes)

print("=== EMBEDDING ===")
print(f"nodes[0,0,:5] = {nodes[0,0,:5].tolist()}")
print(f"nodes[0,3,:5] = {nodes[0,3,:5].tolist()}")
print(f"nodes norm = {mx.sqrt(mx.sum(nodes**2)).item():.6f}")

# Step 2: Check after one P-stream call
state = model._build_state(nodes)
p_out = model.rewrite_step.p_stream(nodes, state.adjacency, state.mask)
mx.eval(p_out)
print(f"\n=== P-STREAM (1st call) ===")
print(f"p_out[0,3,:5] = {p_out[0,3,:5].tolist()}")

# Step 3: D-stream
d_out, new_pred = model.rewrite_step.d_stream(nodes, state.prediction, state.mask)
mx.eval(d_out)
print(f"\n=== D-STREAM ===")
print(f"d_out[0,3,:5] = {d_out[0,3,:5].tolist()}")

# Step 4: I-stream
i_out, new_fw = model.rewrite_step.i_stream(nodes, state.fast_weights, d_out, state.mask)
mx.eval(i_out)
print(f"\n=== I-STREAM ===")
print(f"i_out[0,3,:5] = {i_out[0,3,:5].tolist()}")
print(f"fast_weights norm = {mx.sqrt(mx.sum(new_fw**2)).item():.6f}")

# Step 5: Gate
gate_w, skip = model.rewrite_step.gate(p_out, i_out, d_out, nodes, state.mask)
mx.eval(gate_w)
print(f"\n=== GATE ===")
print(f"gate_weights = {gate_w[0].tolist()}")

# Step 6: Full forward
logits, diag = model(tokens)
last = logits[0, -1, :]
mx.eval(last)
print(f"\n=== FINAL LOGITS ===")
print(f"logits[:5] = {last[:5].tolist()}")
print(f"argmax={mx.argmax(last).item()} max={mx.max(last).item():.4f}")
print(f"Gate P={diag['gate_p'].item():.4f} I={diag['gate_i'].item():.4f} D={diag['gate_d'].item():.4f}")
