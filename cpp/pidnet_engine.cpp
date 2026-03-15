/**
 * PID-Net C++ Inference Engine
 * 
 * Native Metal inference — no Python, no GIL, no garbage collector.
 * Uses MLX C++ API for GPU-accelerated computation on Apple Silicon.
 */

#include "pidnet_model.h"
#include <chrono>
#include <algorithm>
#include <unordered_map>

// Shakespeare char-level vocab (65 chars)
const std::string SHAKESPEARE_CHARS = 
    "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

std::string decode_tokens(const std::vector<int>& tokens, int vocab_size) {
    std::string result;
    if (vocab_size <= 128) {
        // Char-level — use Shakespeare mapping
        for (int t : tokens) {
            if (t >= 0 && t < (int)SHAKESPEARE_CHARS.size()) {
                result += SHAKESPEARE_CHARS[t];
            } else {
                result += '?';
            }
        }
    } else {
        // BPE — just show IDs
        for (size_t i = 0; i < tokens.size(); i++) {
            result += std::to_string(tokens[i]);
            if (i < tokens.size() - 1) result += " ";
        }
    }
    return result;
}

namespace mx = mlx::core;
using namespace std::chrono;

/**
 * Autoregressive generation with frequency-based repetition penalty.
 */
std::vector<int> generate(
    FractalPIDNet& model,
    const std::vector<int>& prompt_tokens,
    int max_new_tokens = 100,
    float temperature = 0.8f,
    int top_k = 50,
    float rep_penalty = 1.5f,
    int penalty_window = 32
) {
    std::vector<int> generated = prompt_tokens;
    
    for (int step = 0; step < max_new_tokens; step++) {
        int seq_len = static_cast<int>(generated.size());
        if (seq_len >= model.max_nodes) break;
        
        // Build input tensor
        auto input = mx::array(generated.data(), {1, seq_len}, mx::int32);
        
        // Forward pass (GPU) — lazy eval, only materialize at sample
        auto logits = model.forward(input);
        
        // Get last token logits: logits is [1, seq_len, vocab]
        auto last_logits = mx::reshape(
            mx::slice(logits, {0, seq_len - 1, 0}, {1, seq_len, model.vocab_size}),
            {model.vocab_size}
        );
        
        // Frequency-based repetition penalty
        if (rep_penalty > 1.0f) {
            int start = std::max(0, seq_len - penalty_window);
            std::unordered_map<int, int> freq;
            for (int i = start; i < seq_len; i++) {
                freq[generated[i]]++;
            }
            
            for (auto& [token_id, count] : freq) {
                if (token_id < model.vocab_size) {
                    float pen = std::pow(rep_penalty, static_cast<float>(count));
                    // Build mask for this token
                    auto mask = mx::arange(model.vocab_size) == mx::array(token_id);
                    auto pen_arr = mx::array(pen);
                    // Apply: divide positive, multiply negative
                    auto is_pos = last_logits > mx::array(0.0f);
                    auto penalized = mx::where(is_pos, last_logits / pen_arr, last_logits * pen_arr);
                    last_logits = mx::where(mask, penalized, last_logits);
                }
            }
        }
        
        // Temperature
        last_logits = last_logits / temperature;
        
        // Top-k
        if (top_k > 0 && top_k < model.vocab_size) {
            auto sorted = mx::sort(last_logits);
            auto threshold = mx::slice(sorted, {model.vocab_size - top_k}, {model.vocab_size - top_k + 1});
            last_logits = mx::where(
                last_logits < threshold,
                mx::array(-1e9f),
                last_logits
            );
        }
        
        // Sample
        auto probs = mx::softmax(last_logits);
        auto log_probs = mx::log(probs + 1e-10f);
        auto next_token = mx::random::categorical(log_probs);
        mx::eval(next_token);
        
        generated.push_back(next_token.item<int>());
    }
    
    return generated;
}

/**
 * Benchmark: measure tokens per second.
 */
void benchmark(FractalPIDNet& model, int n_tokens, int n_runs) {
    std::vector<int> prompt = {0, 1, 2, 3, 4};
    
    // Warmup
    std::cout << "  Warmup..." << std::endl;
    try {
        auto warmup = generate(model, prompt, 5, 0.8f, 50, 1.5f);
        std::cout << "  Warmup OK" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  Warmup failed: " << e.what() << std::endl;
        return;
    }
    
    std::vector<double> times;
    
    for (int run = 0; run < n_runs; run++) {
        auto start = high_resolution_clock::now();
        auto output = generate(model, prompt, n_tokens);
        auto end = high_resolution_clock::now();
        
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times.push_back(ms);
        
        double tps = n_tokens / (ms / 1000.0);
        std::cout << "  Run " << (run + 1) << ": " << ms << "ms | " 
                  << tps << " tok/s" << std::endl;
    }
    
    double avg = 0;
    for (auto t : times) avg += t;
    avg /= times.size();
    double avg_tps = n_tokens / (avg / 1000.0);
    
    std::cout << "\n  ⚡ Average: " << avg << "ms | " << avg_tps << " tok/s" << std::endl;
}

void print_usage() {
    std::cout << "PID-Net C++ Inference Engine\n\n"
              << "Usage:\n"
              << "  pidnet generate --model <path> [options]\n"
              << "  pidnet benchmark --model <path> [options]\n"
              << "\nOptions:\n"
              << "  --model <path>      Weights (.safetensors)\n"
              << "  --tokens <n>        Max new tokens (default: 100)\n"
              << "  --vocab <n>         Vocabulary size (default: 1024)\n"
              << "  --d-model <n>       Model dimension (default: 384)\n"
              << "  --seq-len <n>       Max sequence length (default: 256)\n"
              << "  --levels <n>        Fractal levels (default: 3)\n"
              << "  --runs <n>          Benchmark runs (default: 5)\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    // Parse args
    std::string model_path;
    int max_tokens = 50;
    float temperature = 0.8f;
    int top_k = 50;
    float penalty = 1.5f;
    int vocab = 65;  // Default to Shakespeare char-level
    int d_model = 384;
    int seq_len = 256;
    int levels = 3;
    int runs = 5;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--tokens" && i + 1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (arg == "--vocab" && i + 1 < argc) vocab = std::stoi(argv[++i]);
        else if (arg == "--d-model" && i + 1 < argc) d_model = std::stoi(argv[++i]);
        else if (arg == "--seq-len" && i + 1 < argc) seq_len = std::stoi(argv[++i]);
        else if (arg == "--levels" && i + 1 < argc) levels = std::stoi(argv[++i]);
        else if (arg == "--runs" && i + 1 < argc) runs = std::stoi(argv[++i]);
    }
    
    if (model_path.empty()) {
        std::cerr << "Error: --model required\n";
        return 1;
    }
    
    // Load model
    FractalPIDNet model;
    try {
        model.load(model_path, vocab, d_model, seq_len, 3, 16, levels);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    if (command == "generate") {
        std::cout << "\nGenerating " << max_tokens << " tokens...\n" << std::endl;
        
        std::vector<int> prompt = {0, 1, 2, 3, 4};
        
        auto start = high_resolution_clock::now();
        auto output = generate(model, prompt, max_tokens, temperature, top_k, penalty);
        auto end = high_resolution_clock::now();
        
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double tps = max_tokens / (ms / 1000.0);
        
        std::cout << "Generated " << output.size() << " tokens in " 
                  << ms << "ms (" << tps << " tok/s)\n";
        
        // Decode and print
        std::vector<int> generated_only(output.begin() + prompt.size(), output.end());
        std::string text = decode_tokens(generated_only, model.vocab_size);
        std::cout << "\n--- Generated Text ---\n" << text << "\n---\n" << std::endl;
        
    } else if (command == "benchmark") {
        std::cout << "\n=== PID-Net C++ Inference Benchmark ===" << std::endl;
        std::cout << "  Model: " << model_path << std::endl;
        std::cout << "  Vocab: " << vocab << " | d_model: " << d_model << std::endl;
        std::cout << "  Tokens: " << max_tokens << " | Runs: " << runs << "\n" << std::endl;
        
        benchmark(model, max_tokens, runs);
        
    } else {
        print_usage();
        return 1;
    }
    
    return 0;
}
