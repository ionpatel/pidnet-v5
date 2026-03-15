#pragma once
/**
 * PID-Net Fractal Model — C++ inference using MLX.
 * Weight keys matched to Python FractalPIDNet safetensors format.
 */

#include <mlx/mlx.h>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <unordered_map>

namespace mx = mlx::core;

using WeightMap = std::unordered_map<std::string, mx::array>;

/**
 * Load safetensors and return weight map.
 */
inline WeightMap load_weights(const std::string& path) {
    return mx::load_safetensors(path).first;
}

/**
 * Helper: get weight by key, crash if missing.
 */
inline const mx::array& get_w(const WeightMap& w, const std::string& key) {
    auto it = w.find(key);
    if (it == w.end()) {
        throw std::runtime_error("Missing weight: " + key);
    }
    return it->second;
}

/**
 * Helper: get weight by key, return fallback if missing.
 */
inline mx::array get_w_or(const WeightMap& w, const std::string& key, const mx::array& fallback) {
    auto it = w.find(key);
    return (it != w.end()) ? it->second : fallback;
}

/**
 * LayerNorm using weight/bias from weight map.
 */
inline mx::array layer_norm(const mx::array& x, const mx::array& weight, const mx::array& bias) {
    auto mean = mx::mean(x, std::vector<int>{-1}, true);
    auto var = mx::var(x, std::vector<int>{-1}, true);
    auto normed = (x - mean) / mx::sqrt(var + 1e-5f);
    return normed * weight + bias;
}

/**
 * Fractal PID-Net — direct weight-map forward pass.
 * No class hierarchy — just functions operating on the weight map.
 * This matches the Python model's computation exactly.
 */
class FractalPIDNet {
public:
    int vocab_size = 0;
    int d_model = 0;
    int max_nodes = 0;
    int n_rewrite_steps = 3;
    int chunk_size = 16;
    int n_levels = 3;
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
        for (auto& [name, arr] : w) {
            total += arr.size();
        }
        std::cout << "Loaded PID-Net: " << total << " params, " 
                  << w.size() << " tensors" << std::endl;
    }
    
    /**
     * PID rewrite step using actual weight keys.
     */
    mx::array pid_rewrite(mx::array nodes, mx::array adj, mx::array fast_weights,
                          mx::array prediction) {
        int batch = nodes.shape(0);
        int N = nodes.shape(1);
        int d = nodes.shape(2);
        std::string pre = "rewrite_step.";
        
        // === P-STREAM ===
        auto row_sum = mx::sum(adj, std::vector<int>{-1}, true);
        auto adj_norm = adj / mx::maximum(row_sum, mx::array(1e-8f));
        auto messages = mx::matmul(adj_norm, nodes);
        auto w_msg = get_w(w, pre + "p_stream.message_pass.W_msg.weight");
        auto p_raw = nodes + mx::matmul(messages, mx::transpose(w_msg));
        // P norm + proj
        auto p_normed = layer_norm(p_raw,
            get_w(w, pre + "p_stream.message_pass.norm.weight"),
            get_w(w, pre + "p_stream.message_pass.norm.bias"));
        auto p_out = mx::matmul(p_normed, mx::transpose(get_w(w, pre + "p_stream.proj.weight")))
                     + get_w(w, pre + "p_stream.proj.bias");
        
        // === I-STREAM ===
        auto i_read = mx::matmul(nodes, fast_weights);
        auto i_combined = mx::concatenate({nodes, i_read, nodes * i_read, nodes - i_read}, -1);
        auto i_raw = mx::matmul(i_combined, mx::transpose(get_w(w, pre + "i_stream.W_combine.weight")))
                     + get_w(w, pre + "i_stream.W_combine.bias");
        auto i_normed = layer_norm(i_raw,
            get_w(w, pre + "i_stream.norm.weight"),
            get_w(w, pre + "i_stream.norm.bias"));
        auto i_out = mx::matmul(i_normed, mx::transpose(get_w(w, pre + "i_stream.proj.weight")))
                     + get_w(w, pre + "i_stream.proj.bias");
        
        // === D-STREAM ===
        auto error = nodes - prediction;
        auto d_err = mx::matmul(error, mx::transpose(get_w(w, pre + "d_stream.W_err.weight")))
                     + get_w(w, pre + "d_stream.W_err.bias");
        auto d_normed = layer_norm(d_err,
            get_w(w, pre + "d_stream.norm.weight"),
            get_w(w, pre + "d_stream.norm.bias"));
        auto d_out = mx::matmul(d_normed, mx::transpose(get_w(w, pre + "d_stream.proj.weight")))
                     + get_w(w, pre + "d_stream.proj.bias");
        
        // New prediction (2-layer MLP)
        auto pred_h = mx::matmul(nodes, mx::transpose(get_w(w, pre + "d_stream.predictor.layers.0.weight")))
                      + get_w(w, pre + "d_stream.predictor.layers.0.bias");
        pred_h = mx::maximum(pred_h, mx::array(0.0f)); // ReLU
        auto new_pred = mx::matmul(pred_h, mx::transpose(get_w(w, pre + "d_stream.predictor.layers.2.weight")))
                        + get_w(w, pre + "d_stream.predictor.layers.2.bias");
        
        // === GATE ===
        auto p_mean = mx::mean(p_out, std::vector<int>{1}, true);
        auto i_mean = mx::mean(i_out, std::vector<int>{1}, true);
        auto d_mean = mx::mean(d_out, std::vector<int>{1}, true);
        auto x_mean = mx::mean(nodes, std::vector<int>{1}, true);
        auto gate_in = mx::concatenate({p_mean, i_mean, d_mean, x_mean}, -1);
        
        auto gate_logits = mx::matmul(gate_in, mx::transpose(get_w(w, pre + "gate.W_blend.weight")))
                           + get_w(w, pre + "gate.W_blend.bias");
        auto gates = mx::softmax(gate_logits / 2.0f, -1); // temp=2.0
        gates = mx::maximum(gates, mx::array(0.10f));
        gates = gates / mx::sum(gates, std::vector<int>{-1}, true);
        
        // Weighted combination: gates is [batch, 1, 3]
        auto g_p = mx::slice(gates, {0, 0, 0}, {batch, 1, 1});
        auto g_i = mx::slice(gates, {0, 0, 1}, {batch, 1, 2});
        auto g_d = mx::slice(gates, {0, 0, 2}, {batch, 1, 3});
        
        auto new_nodes = g_p * p_out + g_i * i_out + g_d * d_out;
        
        // Post-norm
        new_nodes = layer_norm(new_nodes,
            get_w(w, pre + "norm.weight"),
            get_w(w, pre + "norm.bias"));
        
        // Energy conservation
        auto norm_before = mx::sqrt(mx::sum(mx::square(nodes), std::vector<int>{-1}, true) + 1e-8f);
        auto norm_after = mx::sqrt(mx::sum(mx::square(new_nodes), std::vector<int>{-1}, true) + 1e-8f);
        auto ratio = mx::clip(norm_before / norm_after, mx::array(0.8f), mx::array(1.2f));
        new_nodes = new_nodes * ratio;
        
        return new_nodes;
    }
    
    /**
     * Forward pass: tokens → last-token logits.
     */
    mx::array forward(const mx::array& tokens) {
        int batch = tokens.shape(0);
        int seq_len = tokens.shape(1);
        
        // Embed
        auto positions = mx::arange(seq_len);
        auto tok_embed = mx::take(get_w(w, "embed.weight"), mx::reshape(tokens, {-1}), 0);
        tok_embed = mx::reshape(tok_embed, {batch, seq_len, d_model});
        auto pos_embed = mx::take(get_w(w, "pos_embed.weight"), positions, 0);
        auto nodes = tok_embed + pos_embed;
        mx::eval(nodes);
        
        // Fractal bottom-up: rewrite at each level
        std::vector<mx::array> level_nodes(n_levels, mx::array(0.0f));
        level_nodes[0] = nodes;
        
        auto current = nodes;
        for (int level = 0; level < n_levels; level++) {
            int N = current.shape(1);
            
            // Causal adjacency
            int k = std::min(8, N);
            auto indices = mx::arange(N);
            auto idx_row = mx::reshape(indices, {N, 1});
            auto idx_col = mx::reshape(indices, {1, N});
            auto dist = mx::astype(idx_row - idx_col, mx::float32);
            auto valid = (dist > mx::array(0.0f)) & (dist <= mx::array(static_cast<float>(k)));
            auto safe_dist = mx::maximum(dist, mx::array(1.0f));
            auto adj = mx::where(valid, mx::array(1.0f) / safe_dist, mx::array(0.0f));
            adj = mx::broadcast_to(mx::expand_dims(adj, 0), {batch, N, N});
            
            auto fast_weights = mx::zeros({batch, d_model, d_model});
            auto prediction = mx::zeros_like(current);
            
            // R rewrite steps (SHARED weights)
            for (int r = 0; r < n_rewrite_steps; r++) {
                current = pid_rewrite(current, adj, fast_weights, prediction);
            }
            
            level_nodes[level] = current;
            
            // Pool to next level (only when evenly divisible)
            if (level < n_levels - 1 && N >= chunk_size * 2 && (N % chunk_size == 0)) {
                int n_chunks = N / chunk_size;
                // Learned pooling
                std::string pool_pre = "pools." + std::to_string(level) + ".";
                auto pooled = mx::reshape(current, {batch, n_chunks, chunk_size, d_model});
                pooled = mx::mean(pooled, std::vector<int>{2}); // Mean pool
                pooled = layer_norm(pooled,
                    get_w(w, pool_pre + "norm.weight"),
                    get_w(w, pool_pre + "norm.bias"));
                pooled = mx::matmul(pooled, mx::transpose(get_w(w, pool_pre + "pool_proj.weight")))
                         + get_w(w, pool_pre + "pool_proj.bias");
                current = pooled;
                level_nodes.push_back(current);
            }
        }
        
        // Readout from level 0
        auto final_nodes = level_nodes[0];
        auto normed = (w.find("readout_norm.weight") != w.end())
            ? layer_norm(final_nodes, get_w(w, "readout_norm.weight"), get_w(w, "readout_norm.bias"))
            : final_nodes;
        
        mx::array logits = tie_weights
            ? mx::matmul(normed, mx::transpose(get_w(w, "embed.weight")))
            : mx::matmul(normed, mx::transpose(get_w(w, "readout.weight")))
              + get_w(w, "readout.bias");
        
        return logits;  // [batch, seq_len, vocab_size]
    }
};
