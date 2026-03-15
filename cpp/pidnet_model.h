#pragma once
/**
 * PID-Net Fractal Model — C++ implementation using MLX.
 * 
 * Mirrors the Python FractalPIDNet architecture.
 * All computation on GPU via MLX Metal backend. No Python.
 */

#include <mlx/mlx.h>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <unordered_map>

namespace mx = mlx::core;

struct GraphState {
    mx::array nodes;        // [batch, N, d]
    mx::array adjacency;    // [batch, N, N]
    mx::array fast_weights; // [batch, d, d]
    mx::array prediction;   // [batch, N, d]
};

struct PIDDiagnostics {
    float p_gate = 0.0f;
    float i_gate = 0.0f;
    float d_gate = 0.0f;
    float pred_error = 0.0f;
    float energy_ratio = 1.0f;
};

/**
 * Load safetensors weights file.
 */
inline std::unordered_map<std::string, mx::array> load_weights(const std::string& path) {
    return mx::load_safetensors(path).first;
}

/**
 * PID Rewrite Step — the core shared rule.
 */
class PIDRewriteStep {
public:
    int d_model = 0;
    float gate_temp = 2.0f;
    float gate_floor = 0.10f;
    float edge_alpha = 0.3f;
    int connect_k = 8;
    
    // Store all weights in a map for flexibility
    std::unordered_map<std::string, mx::array> weights;
    
    void load(const std::unordered_map<std::string, mx::array>& all_weights,
              const std::string& prefix) {
        // Copy relevant weights
        for (auto& [name, arr] : all_weights) {
            if (name.find(prefix) == 0) {
                weights[name.substr(prefix.size())] = arr;
            }
        }
        
        // Get d_model from message passing weight
        if (weights.count("p_stream.message.weight")) {
            d_model = weights["p_stream.message.weight"].shape()[0];
        }
        
        std::cout << "  Loaded PID rewrite step (d=" << d_model 
                  << ", " << weights.size() << " weight tensors)" << std::endl;
    }
    
    mx::array get_w(const std::string& name) {
        auto it = weights.find(name);
        if (it != weights.end()) return it->second;
        std::cerr << "Warning: missing weight '" << name << "'" << std::endl;
        return mx::zeros({d_model, d_model});
    }
    
    /**
     * Forward pass: apply PID rewriting to graph state.
     */
    std::pair<GraphState, PIDDiagnostics> forward(const GraphState& state) {
        auto nodes = state.nodes;
        int batch = nodes.shape(0);
        int N = nodes.shape(1);
        int d = nodes.shape(2);
        
        // === P-STREAM: Message passing ===
        auto row_sum = mx::sum(state.adjacency, std::vector<int>{-1}, true);
        auto adj_norm = state.adjacency / mx::maximum(row_sum, mx::array(1e-8f));
        auto messages = mx::matmul(adj_norm, nodes);
        
        auto w_msg = get_w("p_stream.message.weight");
        auto p_out = nodes + mx::matmul(messages, mx::transpose(w_msg));
        
        // === I-STREAM: Fast weight read ===
        auto i_out = mx::matmul(nodes, state.fast_weights);
        
        // === D-STREAM: Prediction error ===
        auto error = nodes - state.prediction;
        auto d_out = error;
        auto new_pred = nodes;
        
        // === GATE ===
        auto p_mean = mx::mean(p_out, std::vector<int>{1}, true);
        auto i_mean = mx::mean(i_out, std::vector<int>{1}, true);
        auto d_mean = mx::mean(d_out, std::vector<int>{1}, true);
        auto x_mean = mx::mean(nodes, std::vector<int>{1}, true);
        
        auto gate_input = mx::concatenate({p_mean, i_mean, d_mean, x_mean}, -1);
        
        auto gate_logits = weights.count("gate.weight") 
            ? mx::matmul(gate_input, mx::transpose(get_w("gate.weight"))) / gate_temp
            : mx::ones({batch, 1, 3}) / 3.0f;
        
        auto gates = mx::softmax(gate_logits, -1);
        gates = mx::maximum(gates, mx::array(gate_floor));
        auto gate_sum = mx::sum(gates, std::vector<int>{-1}, true);
        gates = gates / gate_sum;
        
        // Extract individual gate values via slicing
        auto g_p = mx::slice(gates, {0, 0, 0}, {batch, 1, 1});
        auto g_i = mx::slice(gates, {0, 0, 1}, {batch, 1, 2});
        auto g_d = mx::slice(gates, {0, 0, 2}, {batch, 1, 3});
        
        auto new_nodes = g_p * p_out + g_i * i_out + g_d * d_out;
        
        // === ENERGY CONSERVATION ===
        auto norm_before = mx::sqrt(mx::sum(mx::square(nodes), std::vector<int>{-1}, true) + 1e-8f);
        auto norm_after = mx::sqrt(mx::sum(mx::square(new_nodes), std::vector<int>{-1}, true) + 1e-8f);
        auto ratio = mx::clip(norm_before / norm_after, mx::array(0.8f), mx::array(1.2f));
        new_nodes = new_nodes * ratio;
        
        // === EDGE EVOLUTION ===
        auto combined = nodes + 0.5f * i_out + 0.5f * d_out;
        auto affinity = mx::matmul(combined, mx::transpose(combined, {0, 2, 1}));
        affinity = affinity / std::sqrt(static_cast<float>(d));
        auto new_edges = mx::sigmoid(affinity);
        
        // Causal mask
        auto causal = mx::tril(mx::ones({N, N}));
        new_edges = new_edges * causal;
        
        // Blend
        auto evolved = (1.0f - edge_alpha) * state.adjacency + edge_alpha * new_edges;
        
        GraphState new_state;
        new_state.nodes = new_nodes;
        new_state.adjacency = evolved;
        new_state.fast_weights = state.fast_weights;
        new_state.prediction = new_pred;
        
        PIDDiagnostics diag;
        return {new_state, diag};
    }
};

/**
 * Fractal PID-Net — full model.
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
    
    mx::array embed_weight;
    mx::array pos_embed_weight;
    mx::array readout_norm_w;
    mx::array readout_norm_b;
    mx::array readout_weight;
    
    PIDRewriteStep rewrite_step;
    
    void load(const std::string& weights_path, int vocab, int d, int max_n,
              int n_steps = 3, int chunk = 16, int levels = 3) {
        vocab_size = vocab;
        d_model = d;
        max_nodes = max_n;
        n_rewrite_steps = n_steps;
        chunk_size = chunk;
        n_levels = levels;
        tie_weights = (vocab > 1000);
        
        auto weights = load_weights(weights_path);
        
        std::cout << "Loading PID-Net (vocab=" << vocab << ", d=" << d 
                  << ", levels=" << levels << ")" << std::endl;
        
        // Load embeddings
        embed_weight = weights.at("embed.weight");
        pos_embed_weight = weights.at("pos_embed.weight");
        readout_norm_w = weights.at("readout_norm.weight");
        readout_norm_b = weights.at("readout_norm.bias");
        
        if (!tie_weights && weights.count("readout.weight")) {
            readout_weight = weights.at("readout.weight");
        }
        
        // Load PID rewrite step
        rewrite_step.load(weights, "rewrite_step.");
        
        int total = 0;
        for (auto& [name, arr] : weights) {
            total += arr.size();
        }
        std::cout << "Total parameters: " << total << std::endl;
    }
    
    /**
     * Forward pass: tokens → logits.
     */
    mx::array forward(const mx::array& tokens) {
        int batch = tokens.shape(0);
        int seq_len = tokens.shape(1);
        
        // Embed
        auto positions = mx::arange(seq_len);
        auto tok_embed = mx::take(embed_weight, tokens.reshape({-1}), 0);
        tok_embed = tok_embed.reshape({batch, seq_len, d_model});
        auto pos_embed = mx::take(pos_embed_weight, positions, 0);
        auto nodes = tok_embed + pos_embed;
        
        // Fractal bottom-up
        for (int level = 0; level < n_levels; level++) {
            int N = nodes.shape(1);
            
            // Build graph state
            GraphState state;
            state.nodes = nodes;
            
            // Initial causal adjacency
            int k = std::min(rewrite_step.connect_k, N);
            auto indices = mx::arange(N);
            auto idx_row = mx::reshape(indices, {N, 1});
            auto idx_col = mx::reshape(indices, {1, N});
            auto dist = mx::astype(idx_row - idx_col, mx::float32);
            auto valid = (dist > mx::array(0.0f)) & (dist <= mx::array(static_cast<float>(k)));
            auto safe_dist = mx::maximum(dist, mx::array(1.0f));
            auto adj = mx::where(valid, mx::array(1.0f) / safe_dist, mx::array(0.0f));
            
            // Broadcast to batch
            adj = mx::broadcast_to(mx::expand_dims(adj, 0), {batch, N, N});
            state.adjacency = adj;
            state.fast_weights = mx::zeros({batch, d_model, d_model});
            state.prediction = mx::zeros_like(nodes);
            
            // R rounds of PID rewriting
            for (int r = 0; r < n_rewrite_steps; r++) {
                auto [new_state, diag] = rewrite_step.forward(state);
                state = new_state;
            }
            
            nodes = state.nodes;
            
            // Pool to next level
            if (level < n_levels - 1 && N >= chunk_size * 2) {
                int n_chunks = N / chunk_size;
                nodes = mx::reshape(nodes, {batch, n_chunks, chunk_size, d_model});
                nodes = mx::mean(nodes, std::vector<int>{2});
            }
        }
        
        // LayerNorm
        auto mean = mx::mean(nodes, std::vector<int>{-1}, true);
        auto var = mx::var(nodes, std::vector<int>{-1}, true);
        auto normed = (nodes - mean) / mx::sqrt(var + 1e-5f);
        normed = normed * readout_norm_w + readout_norm_b;
        
        // Logits
        auto logits = tie_weights
            ? mx::matmul(normed, mx::transpose(embed_weight))
            : mx::matmul(normed, mx::transpose(readout_weight));
        
        return logits;
    }
};
