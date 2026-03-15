/**
 * PID-Net C++ Inference Engine
 * 
 * Native Metal inference — no Python, no GIL, no garbage collector.
 * Uses MLX C++ API for GPU-accelerated computation on Apple Silicon.
 * 
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DMLX_DIR=$(python3 -c "import mlx; print(mlx.__path__[0])")/../..
 *   make -j
 * 
 * Usage:
 *   ./pidnet generate --model weights.safetensors --prompt "Hello"
 *   ./pidnet benchmark --model weights.safetensors --tokens 100
 */

#include "pidnet_model.h"
#include <chrono>
#include <algorithm>
#include <unordered_map>

using namespace std::chrono;

/**
 * Autoregressive token generation with frequency-based repetition penalty.
 * Entirely in C++ — no Python dispatch per token.
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
        if (static_cast<int>(generated.size()) >= model.max_nodes) break;
        
        // Build input tensor
        auto input = mx::array(generated.data(), {1, static_cast<int>(generated.size())}, mx::int32);
        
        // Forward pass (GPU)
        auto logits = model.forward(input);
        
        // Get last token logits: [vocab_size]
        auto last_logits = logits(0, static_cast<int>(generated.size()) - 1);
        
        // Frequency-based repetition penalty
        if (rep_penalty > 1.0f) {
            int window_start = std::max(0, static_cast<int>(generated.size()) - penalty_window);
            std::unordered_map<int, int> freq;
            for (int i = window_start; i < static_cast<int>(generated.size()); i++) {
                freq[generated[i]]++;
            }
            
            for (auto& [token_id, count] : freq) {
                if (token_id < model.vocab_size) {
                    float pen = std::pow(rep_penalty, static_cast<float>(count));
                    auto val = last_logits(token_id);
                    // Positive logits: divide. Negative: multiply.
                    auto is_pos = val > mx::array(0.0f);
                    last_logits = mx::where(
                        mx::arange(model.vocab_size) == token_id,
                        mx::where(is_pos, last_logits / pen, last_logits * pen),
                        last_logits
                    );
                }
            }
        }
        
        // Temperature
        last_logits = last_logits / temperature;
        
        // Top-k
        if (top_k > 0 && top_k < model.vocab_size) {
            auto sorted = mx::sort(last_logits);
            auto threshold = sorted(model.vocab_size - top_k);
            last_logits = mx::where(
                last_logits < threshold, 
                mx::array(-std::numeric_limits<float>::infinity()),
                last_logits
            );
        }
        
        // Sample
        auto probs = mx::softmax(last_logits);
        auto next_token = mx::random::categorical(mx::log(probs + 1e-10f));
        mx::eval(next_token);
        
        generated.push_back(next_token.item<int>());
    }
    
    return generated;
}

/**
 * Benchmark: measure tokens per second.
 */
void benchmark(FractalPIDNet& model, int n_tokens, int n_runs) {
    std::vector<int> prompt = {0, 1, 2, 3, 4};  // Dummy prompt
    
    std::vector<double> times;
    
    // Warmup
    auto warmup = generate(model, prompt, 5);
    
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
    
    // Average
    double avg = 0;
    for (auto t : times) avg += t;
    avg /= times.size();
    double avg_tps = n_tokens / (avg / 1000.0);
    
    std::cout << "\n  Average: " << avg << "ms | " << avg_tps << " tok/s" << std::endl;
}

void print_usage() {
    std::cout << "PID-Net C++ Inference Engine\n"
              << "Usage:\n"
              << "  pidnet generate --model <path> --prompt <text> [options]\n"
              << "  pidnet benchmark --model <path> [options]\n"
              << "\nOptions:\n"
              << "  --model <path>      Model weights (.safetensors)\n"
              << "  --prompt <text>     Prompt text for generation\n"
              << "  --tokens <n>        Max new tokens (default: 100)\n"
              << "  --temperature <f>   Sampling temperature (default: 0.8)\n"
              << "  --top-k <n>         Top-k sampling (default: 50)\n"
              << "  --penalty <f>       Repetition penalty (default: 1.5)\n"
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
    std::string prompt = "The United States";
    int max_tokens = 100;
    float temperature = 0.8f;
    int top_k = 50;
    float penalty = 1.5f;
    int vocab = 1024;
    int d_model = 384;
    int seq_len = 256;
    int levels = 3;
    int runs = 5;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--tokens" && i + 1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc) temperature = std::stof(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) top_k = std::stoi(argv[++i]);
        else if (arg == "--penalty" && i + 1 < argc) penalty = std::stof(argv[++i]);
        else if (arg == "--vocab" && i + 1 < argc) vocab = std::stoi(argv[++i]);
        else if (arg == "--d-model" && i + 1 < argc) d_model = std::stoi(argv[++i]);
        else if (arg == "--seq-len" && i + 1 < argc) seq_len = std::stoi(argv[++i]);
        else if (arg == "--levels" && i + 1 < argc) levels = std::stoi(argv[++i]);
        else if (arg == "--runs" && i + 1 < argc) runs = std::stoi(argv[++i]);
    }
    
    if (model_path.empty()) {
        std::cerr << "Error: --model required" << std::endl;
        return 1;
    }
    
    // Load model
    FractalPIDNet model;
    try {
        model.load(model_path, vocab, d_model, seq_len, 3, 16, levels);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }
    
    if (command == "generate") {
        std::cout << "Generating " << max_tokens << " tokens..." << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        
        // Simple char-level encoding (for testing without tokenizer)
        std::vector<int> prompt_ids;
        for (char c : prompt) {
            prompt_ids.push_back(static_cast<int>(c) % vocab);
        }
        
        auto start = high_resolution_clock::now();
        auto output = generate(model, prompt_ids, max_tokens, temperature, top_k, penalty);
        auto end = high_resolution_clock::now();
        
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double tps = max_tokens / (ms / 1000.0);
        
        std::cout << "\nGenerated " << output.size() << " tokens in " 
                  << ms << "ms (" << tps << " tok/s)" << std::endl;
        
        // Decode (simple — just print token IDs for now)
        std::cout << "Token IDs: ";
        for (int i = prompt_ids.size(); i < static_cast<int>(output.size()); i++) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;
        
    } else if (command == "benchmark") {
        std::cout << "Benchmarking (" << max_tokens << " tokens, " << runs << " runs)..." << std::endl;
        benchmark(model, max_tokens, runs);
        
    } else {
        print_usage();
        return 1;
    }
    
    return 0;
}
