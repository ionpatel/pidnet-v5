/**
 * Genome Prototype — Create, load, and compute with a .genome binary.
 * 
 * This is the first living organism:
 *   - Creates a genome with 6 chromosomes (one per function)
 *   - Each chromosome has genes (weight matrices)
 *   - Transcription factors select which genes to express
 *   - Neurons fire sparsely based on threshold
 *   - Epigenetic state persists across restarts
 * 
 * Build:
 *   c++ -std=c++17 -O2 -o genome_test genome_test.cpp
 * 
 * Run:
 *   ./genome_test create test.genome    # birth
 *   ./genome_test run test.genome       # think
 *   ./genome_test info test.genome      # inspect
 */

#include "genome.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>

// ============================================================
// Helpers
// ============================================================

void random_weights(std::vector<float>& w, int size, float scale) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, scale);
    w.resize(size);
    for (auto& v : w) v = dist(rng);
}

// ============================================================
// CREATE: Birth of an organism
// ============================================================

void create_genome(const char* path) {
    printf("=== GENOME CREATION ===\n");
    
    uint32_t d = 384;
    uint32_t vocab = 4096;
    uint32_t genes_per_chrom = 8;
    uint32_t n_chroms = 6;
    uint32_t n_genes = n_chroms * genes_per_chrom;
    
    // Define chromosomes (one per biological function)
    std::vector<Chromosome> chroms(n_chroms);
    const char* chrom_names[] = {
        "PERCEPTION", "MEMORY", "PREDICTION", 
        "REGULATION", "FOLDING", "OUTPUT"
    };
    ChromosomeFunction funcs[] = {
        ChromosomeFunction::PERCEPTION,
        ChromosomeFunction::MEMORY,
        ChromosomeFunction::PREDICTION,
        ChromosomeFunction::REGULATION,
        ChromosomeFunction::FOLDING,
        ChromosomeFunction::OUTPUT,
    };
    
    for (uint32_t i = 0; i < n_chroms; i++) {
        chroms[i].id = i;
        chroms[i].gene_start = i * genes_per_chrom;
        chroms[i].gene_count = genes_per_chrom;
        chroms[i].function = funcs[i];
        memset(chroms[i]._pad, 0, sizeof(chroms[i]._pad));
    }
    
    // Define genes (each gene = a weight matrix)
    std::vector<GeneDescriptor> gene_descs(n_genes);
    std::vector<std::vector<float>> gene_weights(n_genes);
    
    float scale = 1.0f / std::sqrt((float)d);
    
    for (uint32_t i = 0; i < n_genes; i++) {
        gene_descs[i].id = i;
        gene_descs[i].chromosome = i / genes_per_chrom;
        gene_descs[i].d_in = d;
        gene_descs[i].d_out = d;
        gene_descs[i].gene_type = GeneType::LINEAR;
        gene_descs[i].active = 1;
        gene_descs[i].expression_threshold = 0; // f16 zero
        memset(gene_descs[i].regulates, 0, sizeof(gene_descs[i].regulates));
        memset(gene_descs[i].regulation_weights, 0, sizeof(gene_descs[i].regulation_weights));
        memset(gene_descs[i]._pad, 0, sizeof(gene_descs[i]._pad));
        
        // Initialize gene weights (random for now)
        random_weights(gene_weights[i], d * d, scale);
    }
    
    printf("  Chromosomes: %d\n", n_chroms);
    printf("  Genes: %d (%d per chromosome)\n", n_genes, genes_per_chrom);
    printf("  d_model: %d\n", d);
    printf("  Gene size: %d × %d = %d floats = %lu KB\n",
           d, d, d * d, (d * d * sizeof(float)) / 1024);
    
    bool ok = Genome::create(path, n_chroms, n_genes, d, vocab,
                             chroms, gene_descs, gene_weights);
    
    if (ok) {
        printf("  ✅ Genome created: %s\n", path);
        
        // Verify by opening
        Genome g;
        if (g.open(path)) {
            printf("  ✅ Verified: %d genes, %d chroms, %lu bytes\n",
                   g.n_genes(), g.n_chromosomes(), g.file_size());
            
            // Time gene access
            auto start = std::chrono::high_resolution_clock::now();
            volatile float sum = 0;
            for (int i = 0; i < 1000000; i++) {
                float* data = g.gene_data(i % n_genes);
                sum += data[0];
            }
            auto end = std::chrono::high_resolution_clock::now();
            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count() / 1000000.0;
            printf("  ⚡ Gene access: %.1f ns/lookup (1M lookups)\n", ns);
            
            g.close();
        }
    } else {
        printf("  ❌ Creation failed!\n");
    }
}

// ============================================================
// RUN: The organism thinks
// ============================================================

void run_genome(const char* path) {
    printf("\n=== GENOME EXECUTION ===\n");
    
    Genome g;
    if (!g.open(path)) {
        printf("  ❌ Failed to open %s\n", path);
        return;
    }
    
    uint32_t d = g.d_model();
    printf("  Loaded: %d genes, %d chroms, d=%d\n",
           g.n_genes(), g.n_chromosomes(), d);
    
    // Create a random input (simulating a token embedding)
    std::vector<float> input(d);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : input) v = dist(rng);
    
    // Create neurons
    const int N_NEURONS = 64;
    std::vector<Neuron> neurons(N_NEURONS);
    
    // Assign genes to neurons (each neuron expresses genes from one chromosome)
    for (int i = 0; i < N_NEURONS; i++) {
        uint32_t chrom = i % g.n_chromosomes();
        const Chromosome& chr = g.chromosome(chrom);
        neurons[i].n_genes = std::min((uint8_t)8, (uint8_t)chr.gene_count);
        for (int j = 0; j < neurons[i].n_genes; j++) {
            neurons[i].gene_ids[j] = chr.gene_start + j;
        }
        neurons[i].threshold = 0.5f;
        memset(neurons[i].state, 0, sizeof(neurons[i].state));
        neurons[i].n_connections = 0;
    }
    
    // Wire neurons (sparse random connections)
    for (int i = 0; i < N_NEURONS; i++) {
        int n_conn = 4 + (rng() % 8); // 4-12 connections
        neurons[i].n_connections = std::min(n_conn, 32);
        for (int j = 0; j < neurons[i].n_connections; j++) {
            neurons[i].connections[j] = rng() % N_NEURONS;
        }
    }
    
    printf("\n  --- Thinking (3 rewrite steps) ---\n");
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < 3; step++) {
        int fired = 0;
        int expressed_genes = 0;
        int silenced_genes = 0;
        
        for (int i = 0; i < N_NEURONS; i++) {
            // Transcription: context selects which genes to express
            float expr_weights[8];
            uint32_t chrom = neurons[i].gene_ids[0] / 8;
            g.transcribe(chrom, input.data(), d, expr_weights);
            
            // Translation: express strongest gene → compute
            int best_gene_idx = 0;
            float best_weight = expr_weights[0];
            for (int j = 1; j < neurons[i].n_genes; j++) {
                if (expr_weights[j] > best_weight) {
                    best_weight = expr_weights[j];
                    best_gene_idx = j;
                }
            }
            
            uint32_t gid = neurons[i].gene_ids[best_gene_idx];
            
            // Check epigenetic gating
            if (!g.is_expressed(gid)) {
                silenced_genes++;
                g.record_silence(gid);
                continue;
            }
            expressed_genes++;
            g.record_expression(gid);
            
            // Compute: W × input → state (simplified matmul)
            float* W = g.gene_data(gid);  // NANOSECOND access
            const GeneDescriptor& desc = g.gene(gid);
            
            for (int row = 0; row < (int)desc.d_out && row < 384; row++) {
                float sum = 0;
                for (int col = 0; col < (int)desc.d_in && col < 384; col++) {
                    sum += W[row * desc.d_in + col] * input[col];
                }
                neurons[i].state[row] = sum;
            }
            
            // Sparse firing
            if (neurons[i].should_fire()) {
                fired++;
                
                // Propagate to connected neurons (message passing)
                for (int c = 0; c < neurons[i].n_connections; c++) {
                    uint32_t target = neurons[i].connections[c];
                    // Add state to target's input (simple accumulation)
                    for (int k = 0; k < (int)d; k++) {
                        // Influence connected neurons
                        input[k] += neurons[i].state[k] * 0.01f;
                    }
                }
            }
        }
        
        // Normalize input for next step
        float norm = 0;
        for (int k = 0; k < (int)d; k++) norm += input[k] * input[k];
        norm = std::sqrt(norm / d);
        if (norm > 0) {
            for (int k = 0; k < (int)d; k++) input[k] /= norm;
        }
        
        float firing_rate = (float)fired / N_NEURONS * 100;
        printf("  Step %d: %d/%d neurons fired (%.1f%%), "
               "%d genes expressed, %d silenced\n",
               step, fired, N_NEURONS, firing_rate,
               expressed_genes, silenced_genes);
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(
        total_end - total_start).count();
    
    printf("\n  ⚡ Total compute: %.1f μs (%.0f ns per neuron-step)\n",
           total_us, total_us * 1000.0 / (N_NEURONS * 3));
    
    // Epigenetic state check
    printf("\n  --- Epigenetic State ---\n");
    for (uint32_t i = 0; i < std::min(g.n_genes(), 12u); i++) {
        printf("  Gene %2d: methylation=%.3f expressed=%s\n",
               i, g.expression_level(i),
               g.is_expressed(i) ? "YES" : "NO");
    }
    
    g.close();
    printf("\n  ✅ Organism state saved to disk (mmap writeback)\n");
}

// ============================================================
// INFO: Inspect the organism
// ============================================================

void info_genome(const char* path) {
    printf("\n=== GENOME INFO ===\n");
    
    Genome g;
    if (!g.open(path)) {
        printf("  ❌ Failed to open %s\n", path);
        return;
    }
    
    printf("  File: %s\n", path);
    printf("  Size: %lu bytes (%.1f KB)\n", g.file_size(), g.file_size() / 1024.0);
    printf("  Chromosomes: %d\n", g.n_chromosomes());
    printf("  Genes: %d\n", g.n_genes());
    printf("  d_model: %d\n", g.d_model());
    printf("  Vocab: %d\n", g.vocab_size());
    
    const char* func_names[] = {
        "PERCEPTION", "MEMORY", "PREDICTION",
        "REGULATION", "FOLDING", "OUTPUT", "EMBEDDING"
    };
    
    printf("\n  Chromosomes:\n");
    for (uint32_t i = 0; i < g.n_chromosomes(); i++) {
        const Chromosome& c = g.chromosome(i);
        printf("    [%d] %s: %d genes (start=%d)\n",
               c.id, func_names[(int)c.function],
               c.gene_count, c.gene_start);
    }
    
    printf("\n  Gene Data (first 5 per chromosome):\n");
    for (uint32_t i = 0; i < g.n_genes() && i < 30; i++) {
        const GeneDescriptor& gd = g.gene(i);
        float* data = g.gene_data(i);
        printf("    Gene %2d [chr %d]: %dx%d, type=%d, "
               "first_weight=%.6f, methylation=%.3f\n",
               gd.id, gd.chromosome, gd.d_in, gd.d_out,
               (int)gd.gene_type, data[0], g.expression_level(i));
    }
    
    g.close();
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <create|run|info> <file.genome>\n", argv[0]);
        return 1;
    }
    
    std::string cmd = argv[1];
    const char* path = argv[2];
    
    if (cmd == "create") {
        create_genome(path);
    } else if (cmd == "run") {
        run_genome(path);
    } else if (cmd == "info") {
        info_genome(path);
    } else {
        printf("Unknown command: %s\n", cmd.c_str());
        return 1;
    }
    
    return 0;
}
