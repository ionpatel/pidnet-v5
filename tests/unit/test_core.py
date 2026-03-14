"""
Unit tests for PID-Net v5 core components.

Run: python -m pytest tests/unit/test_core.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx
from pidnet.core.state import GraphState, create_empty_state, add_node, graph_energy
from pidnet.core.message_passing import MessagePassing
from pidnet.streams.p_stream import PStream
from pidnet.streams.i_stream import IStream
from pidnet.streams.d_stream import DStream
from pidnet.gate.pid_gate import PIDGate
from pidnet.model import PIDRewriteStep, PIDGraphNet, count_parameters


def test_empty_state():
    """Create empty state and verify dimensions."""
    state = create_empty_state(batch_size=2, max_nodes=16, d_model=32)
    assert state.nodes.shape == (2, 16, 32)
    assert state.adjacency.shape == (2, 16, 16)
    assert state.fast_weights.shape == (2, 32, 32)
    assert state.mask.shape == (2, 16)
    assert state.n_active.shape == (2,)
    print("✅ test_empty_state passed")


def test_add_node():
    """Add nodes and verify graph grows correctly."""
    state = create_empty_state(1, 16, 32)
    
    # Add 5 nodes
    for i in range(5):
        features = mx.random.normal((1, 32))
        state = add_node(state, features, connect_k=3)
    
    mx.eval(state.n_active)
    assert state.n_active[0].item() == 5
    
    # Check that nodes 0-4 are active
    mx.eval(state.mask)
    for i in range(5):
        assert state.mask[0, i].item() == True
    for i in range(5, 16):
        assert state.mask[0, i].item() == False
    
    print("✅ test_add_node passed")


def test_graph_energy():
    """Verify energy computation doesn't NaN."""
    state = create_empty_state(2, 16, 32)
    for i in range(8):
        features = mx.random.normal((2, 32))
        state = add_node(state, features, connect_k=4)
    
    energy = graph_energy(state)
    mx.eval(energy)
    assert energy.shape == (2,)
    assert not mx.any(mx.isnan(energy)).item()
    print("✅ test_graph_energy passed")


def test_message_passing():
    """Verify message passing produces valid output."""
    mp = MessagePassing(d_model=32)
    
    nodes = mx.random.normal((2, 8, 32))
    adjacency = mx.abs(mx.random.normal((2, 8, 8))) * 0.1
    mask = mx.array([[True]*5 + [False]*3, [True]*8])
    
    output = mp(nodes, adjacency, mask)
    mx.eval(output)
    
    assert output.shape == (2, 8, 32)
    assert not mx.any(mx.isnan(output)).item()
    
    # Inactive nodes should be zero
    assert mx.sum(mx.abs(output[0, 5:])).item() < 1e-6
    
    print("✅ test_message_passing passed")


def test_p_stream():
    """P-stream produces valid output."""
    p = PStream(d_model=32)
    nodes = mx.random.normal((2, 8, 32))
    adj = mx.abs(mx.random.normal((2, 8, 8))) * 0.1
    mask = mx.ones((2, 8), dtype=mx.bool_)
    
    output = p(nodes, adj, mask)
    mx.eval(output)
    assert output.shape == (2, 8, 32)
    assert not mx.any(mx.isnan(output)).item()
    print("✅ test_p_stream passed")


def test_d_stream():
    """D-stream computes prediction error correctly."""
    d = DStream(d_model=32)
    nodes = mx.random.normal((2, 8, 32))
    prediction = mx.random.normal((2, 8, 32))
    mask = mx.ones((2, 8), dtype=mx.bool_)
    
    output, new_pred = d(nodes, prediction, mask)
    mx.eval(output, new_pred)
    
    assert output.shape == (2, 8, 32)
    assert new_pred.shape == (2, 8, 32)
    assert not mx.any(mx.isnan(output)).item()
    
    # When prediction equals actual, error should be near zero
    # (plus stagnation kickout)
    output_zero, _ = d(nodes, nodes, mask)
    mx.eval(output_zero)
    # Error should be smaller than with random prediction
    err_random = mx.mean(mx.abs(output)).item()
    err_zero = mx.mean(mx.abs(output_zero)).item()
    assert err_zero < err_random, f"Zero-error should be smaller: {err_zero} vs {err_random}"
    
    print("✅ test_d_stream passed")


def test_i_stream():
    """I-stream fast weight write and read."""
    i_stream = IStream(d_model=32)
    nodes = mx.random.normal((2, 8, 32))
    fast_weights = mx.zeros((2, 32, 32))
    d_signal = mx.random.normal((2, 8, 32))
    mask = mx.ones((2, 8), dtype=mx.bool_)
    
    output, new_fw = i_stream(nodes, fast_weights, d_signal, mask)
    mx.eval(output, new_fw)
    
    assert output.shape == (2, 8, 32)
    assert new_fw.shape == (2, 32, 32)
    assert not mx.any(mx.isnan(output)).item()
    
    # Fast weights should have been written to (non-zero)
    assert mx.sum(mx.abs(new_fw)).item() > 1e-6, "Fast weights should be non-zero after write"
    
    print("✅ test_i_stream passed")


def test_gate():
    """Gate produces valid blend weights."""
    gate = PIDGate(d_model=32)
    p_out = mx.random.normal((2, 8, 32))
    i_out = mx.random.normal((2, 8, 32))
    d_out = mx.random.normal((2, 8, 32))
    nodes = mx.random.normal((2, 8, 32))
    mask = mx.ones((2, 8), dtype=mx.bool_)
    
    weights, skip = gate(p_out, i_out, d_out, nodes, mask)
    mx.eval(weights, skip)
    
    assert weights.shape == (2, 3)
    assert skip.shape == (2, 1)
    
    # Weights should be positive and approximately sum to 1
    assert mx.all(weights > 0).item(), "Gate weights should be positive"
    weight_sums = mx.sum(weights, axis=-1)
    mx.eval(weight_sums)
    for b in range(2):
        assert abs(weight_sums[b].item() - 1.0) < 0.01, f"Weights should sum to ~1: {weight_sums[b].item()}"
    
    # No gate should be below minimum
    assert mx.all(weights >= 0.04).item(), "Gate values should respect minimum"
    
    print("✅ test_gate passed")


def test_rewrite_step():
    """Full rewrite step produces valid updated state."""
    step = PIDRewriteStep(d_model=32)
    state = create_empty_state(2, 16, 32)
    
    # Add some nodes
    for i in range(8):
        features = mx.random.normal((2, 32))
        state = add_node(state, features, connect_k=4)
    
    new_state, diag = step(state)
    mx.eval(new_state.nodes, new_state.fast_weights)
    
    assert new_state.nodes.shape == state.nodes.shape
    assert not mx.any(mx.isnan(new_state.nodes)).item()
    
    # Diagnostics should exist
    assert 'gate_p' in diag
    assert 'gate_i' in diag
    assert 'gate_d' in diag
    assert 'skip' in diag
    
    print("✅ test_rewrite_step passed")


def test_full_model():
    """Full model forward pass."""
    model = PIDGraphNet(
        vocab_size=65,
        d_model=32,
        max_nodes=24,
        n_rewrite_steps=2,
        connect_k=4,
    )
    
    n_params = count_parameters(model)
    print(f"  Model params: {n_params:,}")
    
    tokens = mx.random.randint(0, 65, (2, 16))
    logits, diag = model(tokens)
    mx.eval(logits)
    
    assert logits.shape == (2, 16, 65), f"Expected (2, 16, 65), got {logits.shape}"
    assert not mx.any(mx.isnan(logits)).item()
    
    print("✅ test_full_model passed")


def test_generation_no_crash():
    """Generation doesn't crash (doesn't test quality)."""
    model = PIDGraphNet(
        vocab_size=65,
        d_model=32,
        max_nodes=32,
        n_rewrite_steps=2,
        connect_k=4,
    )
    
    prompt = mx.array([[0, 1, 2, 3, 4]])
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    mx.eval(generated)
    
    assert generated.shape[0] == 15  # 5 prompt + 10 generated
    print("✅ test_generation_no_crash passed")


if __name__ == "__main__":
    tests = [
        test_empty_state,
        test_add_node,
        test_graph_energy,
        test_message_passing,
        test_p_stream,
        test_d_stream,
        test_i_stream,
        test_gate,
        test_rewrite_step,
        test_full_model,
        test_generation_no_crash,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*40}")
