#pragma once
/**
 * PID-Net Fractal Model — C++ inference using MLX.
 * Faithfully matches Python FractalPIDNet forward pass.
 */

#include <mlx/mlx.h>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <unordered_map>

namespace mx = mlx::core;
using WeightMap = std::unordered_map<std::string, mx::array>;

inline WeightMap load_weights(const std::string& path) {
    return mx::load_safetensors(path).first;
}

inline const mx::array& W(const WeightMap& w, const std::string& key) {
    auto it = w.find(key);
    if (it == w.end()) throw std::runtime_error("Missing weight: " + key);
    return it->second;
}

inline mx::array layer_norm(const mx::array& x, const mx::array& weight, const mx::array& bias) {
    auto mean = mx::mean(x, std::vector<int>{-1}, true);
    auto var = mx::var(x, std::vector<int>{-1}, true);
    return (x - mean) / mx::sqrt(var + 1e-5f) * weight + bias;
}

// Linear layer: x @ W^T + b
inline mx::array linear(const WeightMap& w, const std::string& prefix, const mx::array& x) {
    auto out = mx::matmul(x, mx::transpose(W(w, prefix + ".weight")));
    auto it = w.find(prefix + ".bias");
    if (it != w.end()) out = out + it->second;
    return out;
}

// Linear layer without bias
inline mx::array linear_nb(const WeightMap& w, const std::string& prefix, const mx::array& x) {
    return mx::matmul(x, mx::transpose(W(w, prefix + ".weight")));
}

inline mx::array ln(const WeightMap& w, const std::string& prefix, const mx::array& x) {
    return layer_norm(x, W(w, prefix + ".weight"), W(w, prefix + ".bias"));
}

class FractalPIDNet {
public:
    int vocab_size = 0;
    int d_model = 0;
    int max_nodes = 0;
    int n_rewrite_steps = 3;
    int chunk_size = 16;
    int n_levels = 3;
    int connect_k = 8;
    bool tie_weights = false;
    WeightMap w;
    
    void load(const std::string& path, int vocab, int d, int max_n,
              int n_steps = 3, int chunk = 16, int levels = 3) {
        vocab_size = vocab;
        d_model = d;
        max_nodes = max_n;
        n_rewrite_steps = n_steps;
        chunk_size = chunk;
        n_levels = levels;
        tie_weights = (vocab > 1000);
        
        w = load_weights(path);
        
        int total = 0;
        for (auto& [name, arr] : w) total += arr.size();
        std::cout << "Loaded PID-Net: " << total << " params, " 
                  << w.size() << " tensors" << std::endl;
    }
    
    // Adjacency cache: avoid rebuilding for same sequence length
    std::unordered_map<int, mx::array> adj_cache;
    
    /**
     * Build causal adjacency with forward AND backward edges.
     * Cached by sequence length.
     */
    mx::array build_adjacency(int N, int batch) {
        auto it = adj_cache.find(N);
        if (it != adj_cache.end()) {
            return mx::broadcast_to(mx::expand_dims(it->second, 0), {batch, N, N});
        }
        int k = std::min(connect_k, N);
        auto indices = mx::arange(N);
        auto idx_row = mx::reshape(indices, {N, 1});
        auto idx_col = mx::reshape(indices, {1, N});
        auto dist = mx::astype(idx_row - idx_col, mx::float32);
        
        // Forward: j→i where dist > 0 && dist <= k, weight = 1/dist
        auto valid_fwd = (dist > mx::array(0.0f)) & (dist <= mx::array(static_cast<float>(k)));
        auto safe_dist = mx::maximum(dist, mx::array(1.0f));
        auto strength_fwd = mx::where(valid_fwd, mx::array(1.0f) / safe_dist, mx::array(0.0f));
        
        // Backward: i→j where dist < 0 && -dist <= k, weight = 0.5/(-dist)
        auto neg_dist = mx::array(0.0f) - dist;
        auto valid_back = (dist < mx::array(0.0f)) & (neg_dist <= mx::array(static_cast<float>(k)));
        auto safe_neg = mx::maximum(neg_dist, mx::array(1.0f));
        auto strength_back = mx::where(valid_back, mx::array(0.5f) / safe_neg, mx::array(0.0f));
        
        auto adj = strength_fwd + strength_back;
        
        // Causal mask
        auto causal = mx::tril(mx::ones({N, N}));
        adj = adj * causal;
        
        // Cache the 2D adjacency (without batch dim)
        mx::eval(adj);
        adj_cache.insert_or_assign(N, adj);
        return mx::broadcast_to(mx::expand_dims(adj, 0), {batch, N, N});
    }
    
    /**
     * P-Stream: Message passing on graph.
     */
    mx::array p_stream(const mx::array& nodes, const mx::array& adj) {
        std::string pre = "rewrite_step.p_stream.";
        // 1. Transform nodes FIRST
        auto transformed = linear_nb(w, pre + "message_pass.W_msg", nodes);
        // 2. Aggregate transformed features via adjacency
        auto row_sum = mx::sum(adj, std::vector<int>{-1}, true) + 1e-8f;
        auto adj_norm = adj / row_sum;
        auto messages = mx::matmul(adj_norm, transformed);
        // 3. Residual + norm
        auto perceived = ln(w, pre + "message_pass.norm", nodes + messages);
        // 4. Project with GELU activation
        auto output = linear(w, pre + "proj", perceived);
        // GELU approximation: x * sigmoid(1.702 * x)
        output = output * mx::sigmoid(output * 1.702f);
        return output;
    }
    
    /**
     * I-Stream: Fast weight associative memory with D→I coupling.
     */
    void i_stream(
        const mx::array& nodes,
        const mx::array& fast_weights,
        const mx::array& d_signal,
        mx::array& output,
        mx::array& new_fw_out
    ) {
        std::string pre = "rewrite_step.i_stream.";
        int batch = nodes.shape(0);
        int N = nodes.shape(1);
        
        // Node summary (mean over sequence)
        auto node_summary = mx::mean(nodes, std::vector<int>{1}); // [batch, d]
        
        // Keys and values from summary
        auto keys = linear_nb(w, pre + "W_key", node_summary);    // [batch, d]
        auto values = linear_nb(w, pre + "W_value", node_summary); // [batch, d]
        
        // D→I coupling: write strength from prediction error
        auto d_summary = mx::mean(d_signal, std::vector<int>{1}); // [batch, d]
        auto eta = mx::sigmoid(linear(w, pre + "W_eta", d_summary)); // [batch, n_timescales]
        
        // Outer product: keys ⊗ values
        auto k_exp = mx::expand_dims(keys, -1);   // [batch, d, 1]
        auto v_exp = mx::expand_dims(values, -2);  // [batch, 1, d]
        auto outer = k_exp * v_exp;                 // [batch, d, d]
        
        // Multi-timescale decay
        auto avg_eta = mx::mean(eta, std::vector<int>{-1});  // [batch]
        avg_eta = mx::reshape(avg_eta, {batch, 1, 1});
        auto decay_rates = W(w, pre + "decay_rates");
        float avg_decay = mx::mean(decay_rates).item<float>();
        
        auto new_fw = avg_decay * fast_weights + avg_eta * outer;
        
        // Read: query from current nodes
        auto queries = linear_nb(w, pre + "W_query", nodes); // [batch, N, d]
        auto retrieved = mx::matmul(queries, mx::transpose(new_fw, {0, 2, 1})); // [batch, N, d]
        
        // Project and normalize
        output = ln(w, pre + "norm", linear(w, pre + "proj", retrieved));
        new_fw_out = new_fw;
    }
    
    /**
     * D-Stream: Prediction error (surprise).
     */
    void d_stream(
        const mx::array& nodes,
        const mx::array& prediction,
        mx::array& d_out,
        mx::array& new_pred
    ) {
        std::string pre = "rewrite_step.d_stream.";
        
        auto epsilon = nodes - prediction;
        
        // Stagnation kickout (simplified — full version has diversity detection)
        auto kickout = linear(w, pre + "kickout_proj", nodes);
        auto kickout_strength = mx::sigmoid(linear(w, pre + "kickout_gate", nodes));
        // Diversity: cosine similarity between consecutive nodes
        int N = nodes.shape(1);
        auto shifted = mx::concatenate({mx::zeros({nodes.shape(0), 1, nodes.shape(2)}), 
                                         mx::slice(nodes, {0,0,0}, {nodes.shape(0), N-1, nodes.shape(2)})}, 1);
        auto nodes_norm = mx::sqrt(mx::sum(mx::square(nodes), std::vector<int>{-1}, true) + 1e-8f);
        auto shifted_norm = mx::sqrt(mx::sum(mx::square(shifted), std::vector<int>{-1}, true) + 1e-8f);
        auto neighbor_cos = mx::sum(nodes * shifted, std::vector<int>{-1}, true) / (nodes_norm * shifted_norm);
        auto stagnation = mx::sigmoid((neighbor_cos - 0.999f) * 500.0f);
        auto gated_kickout = stagnation * kickout_strength * kickout;
        auto diff = nodes - shifted;
        auto diversity_penalty = linear_nb(w, pre + "diversity_proj", diff) * stagnation;
        epsilon = epsilon + gated_kickout + diversity_penalty * 0.5f;
        
        // W_err → proj → norm (correct order!)
        auto d_signal = linear(w, pre + "W_err", epsilon);
        d_out = ln(w, pre + "norm", linear(w, pre + "proj", d_signal));
        
        // New prediction (2-layer MLP with GELU, not ReLU!)
        auto pred_h = linear(w, pre + "predictor.layers.0", nodes);
        pred_h = pred_h * mx::sigmoid(pred_h * 1.702f);  // GELU
        new_pred = linear(w, pre + "predictor.layers.2", pred_h);
    }
    
    /**
     * Edge Evolution: dynamic topology based on I and D signals.
     * The graph BREATHES — edges born, strengthened, weakened based on PID signals.
     */
    mx::array evolve_edges(
        const mx::array& adjacency, const mx::array& nodes,
        const mx::array& i_signal, const mx::array& d_signal
    ) {
        std::string pre = "rewrite_step.edge_evolver.";
        int batch = nodes.shape(0);
        int N = nodes.shape(1);
        int d = nodes.shape(2);
        
        // Modulate features with I and D signals
        auto i_feat = linear_nb(w, pre + "W_i_edge", i_signal);
        auto d_feat = linear_nb(w, pre + "W_d_edge", d_signal);
        auto combined = nodes + i_feat * 0.5f + d_feat * 0.5f;
        
        // Pairwise dot-product scores / sqrt(d)
        auto scores = mx::matmul(combined, mx::transpose(combined, {0, 2, 1}));
        scores = scores / std::sqrt(static_cast<float>(d));
        
        // Causal mask + no self-loops
        auto causal = mx::tril(mx::ones({N, N}));
        auto eye = mx::eye(N);
        scores = scores * causal * (mx::array(1.0f) - eye) + eye * (-1e9f);
        
        // Sigmoid → edge probabilities
        auto new_adj = mx::sigmoid(scores);
        new_adj = new_adj * causal;
        
        // Top-k sparsification (keep strongest 16 incoming edges per node)
        int top_k = std::min(16, N);
        if (top_k < N) {
            auto rank = mx::argsort(mx::argsort(mx::array(0.0f) - new_adj, -1), -1);
            auto topk_mask = mx::astype(rank < mx::array(top_k), mx::float32);
            new_adj = new_adj * topk_mask;
        }
        
        // Blend: 70% old + 30% new (don't change too fast)
        return adjacency * 0.7f + new_adj * 0.3f;
    }
    
    /**
     * PID Gate: blend P/I/D with temperature + min floor.
     */
    mx::array gate_blend(
        const mx::array& p_out, const mx::array& i_out,
        const mx::array& d_out, const mx::array& nodes
    ) {
        std::string pre = "rewrite_step.gate.";
        int batch = nodes.shape(0);
        
        auto p_mean = mx::mean(p_out, std::vector<int>{1}); // [batch, d]
        auto i_mean = mx::mean(i_out, std::vector<int>{1});
        auto d_mean = mx::mean(d_out, std::vector<int>{1});
        auto x_mean = mx::mean(nodes, std::vector<int>{1});
        
        auto gate_in = mx::concatenate({p_mean, i_mean, d_mean, x_mean}, -1); // [batch, 4d]
        
        // Softmax with temperature 2.0
        auto logits = linear(w, pre + "W_blend", gate_in) / 2.0f; // [batch, 3]
        auto gates = mx::softmax(logits, -1);
        
        // Min gate floor: gates * 0.7 + 0.1 (matches Python exactly)
        gates = gates * 0.7f + 0.1f;
        
        // Skip is disabled (always 0) — matches Python skip_enabled=False
        // new_nodes = 0*nodes + 1*norm(blended) = norm(blended)
        
        // Expand for broadcasting: [batch, 3] → [batch, 1, 1] per gate
        auto gp = mx::reshape(mx::slice(gates, {0, 0}, {batch, 1}), {batch, 1, 1});
        auto gi = mx::reshape(mx::slice(gates, {0, 1}, {batch, 2}), {batch, 1, 1});
        auto gd = mx::reshape(mx::slice(gates, {0, 2}, {batch, 3}), {batch, 1, 1});
        
        auto blended = gp * p_out + gi * i_out + gd * d_out;
        return ln(w, "rewrite_step.norm", blended);
    }
    
    /**
     * One PID rewrite step. Updates nodes, fast_weights, prediction.
     */
    struct RewriteState {
        mx::array nodes = mx::array(0.0f);
        mx::array adjacency = mx::array(0.0f);
        mx::array fast_weights = mx::array(0.0f);
        mx::array prediction = mx::array(0.0f);
    };
    
    RewriteState pid_rewrite(
        const mx::array& nodes, const mx::array& adj,
        const mx::array& fast_weights, const mx::array& prediction
    ) {
        auto p_out = p_stream(nodes, adj);
        
        mx::array d_out = mx::array(0.0f), new_pred = mx::array(0.0f);
        d_stream(nodes, prediction, d_out, new_pred);
        
        mx::array i_out = mx::array(0.0f), new_fw = mx::array(0.0f);
        i_stream(nodes, fast_weights, d_out, i_out, new_fw);
        
        auto new_nodes = gate_blend(p_out, i_out, d_out, nodes);
        
        // Edge evolution — the graph topology changes
        auto new_adj = evolve_edges(adj, new_nodes, i_out, d_out);
        
        RewriteState result;
        result.nodes = new_nodes;
        result.adjacency = new_adj;
        result.fast_weights = new_fw;
        result.prediction = new_pred;
        return result;
    }
    
    // === KV CACHE: Incremental inference ===
    // Cache the rewritten nodes from previous steps so we don't
    // recompute the entire sequence for each new token.
    struct InferenceCache {
        mx::array nodes = mx::array(0.0f);       // [1, cached_len, d]
        mx::array fast_weights = mx::array(0.0f); // [1, d, d]
        mx::array prediction = mx::array(0.0f);   // [1, cached_len, d]
        int cached_len = 0;
        bool valid = false;
    };
    InferenceCache cache;
    
    void clear_cache() { cache.valid = false; cache.cached_len = 0; }
    
    /**
     * Incremental forward: only compute new token, reuse cached state.
     * Falls back to full forward when cache is invalid or seq changes.
     */
    mx::array forward_incremental(const mx::array& tokens) {
        int seq_len = tokens.shape(1);
        
        // If cache valid and we're just adding one token, use incremental
        if (cache.valid && seq_len == cache.cached_len + 1) {
            return forward_one_token(tokens);
        }
        
        // Otherwise do full forward and populate cache
        auto logits = forward(tokens);
        return logits;
    }
    
    /**
     * Process only the new (last) token using cached state.
     */
    mx::array forward_one_token(const mx::array& tokens) {
        int batch = tokens.shape(0);
        int seq_len = tokens.shape(1);
        
        // Embed only the new token
        auto new_pos = mx::array(seq_len - 1);
        auto new_tok = mx::slice(tokens, {0, seq_len - 1}, {1, seq_len});
        auto new_embed = mx::take(W(w, "embed.weight"), mx::reshape(new_tok, {-1}), 0);
        new_embed = mx::reshape(new_embed, {1, 1, d_model});
        auto new_pos_embed = mx::take(W(w, "pos_embed.weight"), 
                                       mx::reshape(new_pos, {1}), 0);
        auto new_node = new_embed + new_pos_embed;
        
        // Concatenate with cached nodes
        auto all_nodes = mx::concatenate({cache.nodes, new_node}, 1);
        auto all_pred = mx::concatenate({cache.prediction, mx::zeros({1, 1, d_model})}, 1);
        
        // Run rewrite on full sequence (but adjacency only needs one new row/col)
        auto adj = build_adjacency(seq_len, batch);
        auto fast_weights = cache.fast_weights;
        auto prediction = all_pred;
        auto current = all_nodes;
        
        auto norm_before = mx::sqrt(mx::sum(mx::square(current), std::vector<int>{-1}, true) + 1e-8f);
        
        for (int r = 0; r < n_rewrite_steps; r++) {
            auto state = pid_rewrite(current, adj, fast_weights, prediction);
            current = state.nodes;
            adj = state.adjacency;
            fast_weights = state.fast_weights;
            prediction = state.prediction;
        }
        
        auto norm_after = mx::sqrt(mx::sum(mx::square(current), std::vector<int>{-1}, true) + 1e-8f);
        auto ratio = mx::clip(norm_before / norm_after, mx::array(0.8f), mx::array(1.2f));
        current = current * ratio;
        
        // Update cache
        cache.nodes = current;
        cache.fast_weights = fast_weights;
        cache.prediction = prediction;
        cache.cached_len = seq_len;
        cache.valid = true;
        
        // Readout
        auto normed = (w.find("readout_norm.weight") != w.end())
            ? ln(w, "readout_norm", current)
            : current;
        
        auto logits = tie_weights
            ? mx::matmul(normed, mx::transpose(W(w, "embed.weight")))
            : linear(w, "readout", normed);
        
        return logits;
    }
    
    /**
     * Full forward pass: tokens → logits [batch, seq, vocab].
     */
    mx::array forward(const mx::array& tokens) {
        int batch = tokens.shape(0);
        int seq_len = tokens.shape(1);
        
        // Embed
        auto positions = mx::arange(seq_len);
        auto tok_embed = mx::take(W(w, "embed.weight"), mx::reshape(tokens, {-1}), 0);
        tok_embed = mx::reshape(tok_embed, {batch, seq_len, d_model});
        auto pos_embed = mx::take(W(w, "pos_embed.weight"), positions, 0);
        auto nodes = tok_embed + pos_embed;
        
        // === BOTTOM-UP: Rewrite at each level, then pool ===
        std::vector<mx::array> level_nodes(n_levels, mx::array(0.0f));
        level_nodes[0] = nodes;
        
        auto current = nodes;
        for (int level = 0; level < n_levels; level++) {
            int N = current.shape(1);
            auto adj = build_adjacency(N, batch);
            auto fast_weights = mx::zeros({batch, d_model, d_model});
            auto prediction = mx::zeros_like(current);
            
            // Norm before rewriting
            auto norm_before = mx::sqrt(mx::sum(mx::square(current), std::vector<int>{-1}, true) + 1e-8f);
            
            // R rewrite steps (SHARED weights, evolving edges)
            for (int r = 0; r < n_rewrite_steps; r++) {
                auto state = pid_rewrite(current, adj, fast_weights, prediction);
                current = state.nodes;
                adj = state.adjacency;  // edges evolve!
                fast_weights = state.fast_weights;
                prediction = state.prediction;
            }
            
            // Norm preservation
            auto norm_after = mx::sqrt(mx::sum(mx::square(current), std::vector<int>{-1}, true) + 1e-8f);
            auto ratio = mx::clip(norm_before / norm_after, mx::array(0.8f), mx::array(1.2f));
            current = current * ratio;
            
            level_nodes[level] = current;
            
            // Cache level 0 state for incremental inference
            if (level == 0) {
                cache.nodes = current;
                cache.fast_weights = fast_weights;
                cache.prediction = prediction;
                cache.cached_len = seq_len;
                cache.valid = true;
            }
            
            // Pool to next level
            if (level < n_levels - 1 && N >= chunk_size * 2 && (N % chunk_size == 0)) {
                int n_chunks = N / chunk_size;
                std::string pp = "pools." + std::to_string(level) + ".";
                auto pooled = mx::reshape(current, {batch, n_chunks, chunk_size, d_model});
                pooled = mx::mean(pooled, std::vector<int>{2});
                pooled = ln(w, pp + "norm", pooled);
                pooled = linear(w, pp + "pool_proj", pooled);
                current = pooled;
                level_nodes.push_back(current);
            }
        }
        
        // === TOP-DOWN: Broadcast higher-level context ===
        int max_broadcast = std::min(static_cast<int>(level_nodes.size()) - 1, n_levels - 1);
        for (int level = max_broadcast; level > 0; level--) {
            auto higher = level_nodes[level];
            auto lower = level_nodes[level - 1];
            int N = lower.shape(1);
            int n_chunks = higher.shape(1);
            
            std::string bp = "broadcasts." + std::to_string(level - 1) + ".";
            
            // Project chunk features
            auto chunk_ctx = linear(w, bp + "broadcast_proj", higher); // [batch, n_chunks, d]
            
            // Expand: repeat each chunk for chunk_size tokens
            // [batch, n_chunks, 1, d] → [batch, n_chunks, chunk_size, d] → [batch, N_trunc, d]
            auto expanded = mx::repeat(mx::expand_dims(chunk_ctx, 2), chunk_size, 2);
            expanded = mx::reshape(expanded, {batch, n_chunks * chunk_size, d_model});
            
            // Pad or truncate to match lower level
            if (expanded.shape(1) < N) {
                auto padding = mx::zeros({batch, N - expanded.shape(1), d_model});
                expanded = mx::concatenate({expanded, padding}, 1);
            } else if (expanded.shape(1) > N) {
                expanded = mx::slice(expanded, {0, 0, 0}, {batch, N, d_model});
            }
            
            // Gated addition
            auto gate_in = mx::concatenate({lower, expanded}, -1); // [batch, N, 2d]
            auto gate_val = mx::sigmoid(linear(w, bp + "gate", gate_in)); // [batch, N, 1]
            level_nodes[level - 1] = lower + gate_val * expanded;
        }
        
        // Cache is populated inside the level-0 loop below
        
        // === READOUT ===
        auto final_nodes = level_nodes[0];
        auto normed = (w.find("readout_norm.weight") != w.end())
            ? ln(w, "readout_norm", final_nodes)
            : final_nodes;
        
        auto logits = tie_weights
            ? mx::matmul(normed, mx::transpose(W(w, "embed.weight")))
            : linear(w, "readout", normed);
        
        return logits;
    }
};
