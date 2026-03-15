#pragma once
/**
 * Genome Binary Format — DNA-structured neural storage.
 * 
 * The .genome file IS the living organism:
 *   - mmap'd for nanosecond access
 *   - Zero parsing, zero copying
 *   - Epigenetic state persists through mmap writeback
 *   - Kill + restart = organism wakes up where it was
 *   - Copy file = clone organism
 */

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <functional>

// ============================================================
// Binary Format Structures (cache-line aligned = 64 bytes each)
// ============================================================

static constexpr uint32_t GENOME_MAGIC = 0x4F4E4547; // "GENO"
static constexpr uint32_t GENOME_VERSION = 1;
static constexpr size_t CACHE_LINE = 64;

enum class ChromosomeFunction : uint32_t {
    PERCEPTION  = 0,  // P-stream analog: process current input
    MEMORY      = 1,  // I-stream analog: fast weight associative memory
    PREDICTION  = 2,  // D-stream analog: prediction error
    REGULATION  = 3,  // Gene regulatory network
    FOLDING     = 4,  // Topology generation
    OUTPUT      = 5,  // Readout
    EMBEDDING   = 6,  // Token → vector
};

enum class GeneType : uint8_t {
    LINEAR      = 0,  // W × x + b
    OUTER_PROD  = 1,  // outer(a, b) — fast weight write
    GATE        = 2,  // sigmoid/softmax selection
    NORM        = 3,  // layer norm params
    ACTIVATION  = 4,  // learned activation
    ATTENTION   = 5,  // Q/K/V projections
    CUSTOM      = 6,  // gene-defined computation
};

struct __attribute__((packed, aligned(64))) GenomeHeader {
    uint32_t magic;           // "GENO" = 0x4F4E4547
    uint32_t version;
    uint32_t n_chromosomes;
    uint32_t n_genes;
    uint32_t d_model;
    uint32_t vocab_size;
    uint64_t genome_length;   // total file size
    uint64_t gene_data_offset;
    uint64_t epigenetic_offset;
    uint64_t regulation_offset;
    uint64_t checksum;
    uint8_t  _pad[8];         // pad to 64 bytes
};
static_assert(sizeof(GenomeHeader) == 64, "Header must be 1 cache line");

struct __attribute__((packed, aligned(64))) Chromosome {
    uint32_t id;
    uint32_t gene_start;      // index of first gene
    uint32_t gene_count;
    ChromosomeFunction function;
    uint64_t data_offset;     // byte offset to chromosome's gene data
    uint64_t data_size;
    uint8_t  _pad[32];       // pad to 64 bytes
};
static_assert(sizeof(Chromosome) == 64, "Chromosome must be 1 cache line");

struct __attribute__((packed, aligned(64))) GeneDescriptor {
    uint32_t id;
    uint32_t chromosome;      // which chromosome this gene belongs to
    uint64_t data_offset;     // byte offset to weight data
    uint32_t data_size;       // size in bytes
    uint16_t d_in;
    uint16_t d_out;
    GeneType gene_type;
    uint8_t  active;          // 1 = can be expressed, 0 = silenced
    uint16_t expression_threshold; // f16: minimum transcription factor to express
    uint32_t regulates[4];    // gene IDs this gene regulates (regulatory network)
    uint16_t regulation_weights[4]; // f16: how strongly it regulates each target
    uint8_t  _pad[10];       // pad to 64 bytes
};
static_assert(sizeof(GeneDescriptor) == 64, "Gene must be 1 cache line");

// ============================================================
// Epigenetic State — persistent meta-memory
// ============================================================

struct EpigeneticState {
    float* methylation;       // [n_genes] — accessibility level per gene
    float* expression_history; // [n_genes] — running avg of expression frequency
    uint64_t total_steps;     // how many compute steps this organism has lived
};

// ============================================================
// Neuron — the compute unit
// ============================================================

struct Neuron {
    uint32_t gene_ids[8];     // genes this neuron can express (max 8)
    uint8_t  n_genes;         // actual number of genes
    float    state[384];      // current activation
    float    threshold;       // firing threshold (sparse activation)
    uint32_t connections[32]; // outgoing synapse targets (neuron IDs)
    uint8_t  n_connections;
    
    inline float energy() const {
        float e = 0;
        for (int i = 0; i < 384; i++) e += state[i] * state[i];
        return e;
    }
    
    inline bool should_fire() const {
        return energy() > threshold * threshold * 384;
    }
};

// ============================================================
// The Genome — the living organism
// ============================================================

class Genome {
    int fd_ = -1;
    void* base_ = nullptr;
    size_t file_size_ = 0;
    
    // Direct pointers into mmap'd region (NANOSECOND access)
    GenomeHeader*   header_   = nullptr;
    Chromosome*     chroms_   = nullptr;
    GeneDescriptor* genes_    = nullptr;
    char*           gene_data_ = nullptr;
    float*          epi_methylation_ = nullptr;
    float*          epi_history_ = nullptr;

public:
    // ---- Lifecycle ----
    
    ~Genome() { close(); }
    
    /**
     * Open an existing .genome file (the organism wakes up).
     */
    bool open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDWR);
        if (fd_ < 0) return false;
        
        struct stat st;
        fstat(fd_, &st);
        file_size_ = st.st_size;
        
        // mmap with read/write — changes persist to disk
        base_ = mmap(nullptr, file_size_, 
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) { ::close(fd_); return false; }
        
        // Wire up pointers (NO parsing — just pointer arithmetic)
        header_ = reinterpret_cast<GenomeHeader*>(base_);
        if (header_->magic != GENOME_MAGIC) { close(); return false; }
        
        chroms_ = reinterpret_cast<Chromosome*>(
            static_cast<char*>(base_) + sizeof(GenomeHeader));
        
        genes_ = reinterpret_cast<GeneDescriptor*>(
            static_cast<char*>(base_) + sizeof(GenomeHeader) + 
            header_->n_chromosomes * sizeof(Chromosome));
        
        gene_data_ = static_cast<char*>(base_) + header_->gene_data_offset;
        
        if (header_->epigenetic_offset > 0) {
            epi_methylation_ = reinterpret_cast<float*>(
                static_cast<char*>(base_) + header_->epigenetic_offset);
            epi_history_ = epi_methylation_ + header_->n_genes;
        }
        
        return true;
    }
    
    void close() {
        if (base_ && base_ != MAP_FAILED) {
            msync(base_, file_size_, MS_SYNC); // flush changes to disk
            munmap(base_, file_size_);
        }
        if (fd_ >= 0) ::close(fd_);
        base_ = nullptr;
        fd_ = -1;
    }
    
    /**
     * Create a new .genome file (birth of an organism).
     */
    static bool create(const std::string& path,
                       uint32_t n_chromosomes,
                       uint32_t n_genes,
                       uint32_t d_model,
                       uint32_t vocab_size,
                       const std::vector<Chromosome>& chroms,
                       const std::vector<GeneDescriptor>& gene_descs,
                       const std::vector<std::vector<float>>& gene_weights) {
        
        // Calculate offsets
        size_t header_size = sizeof(GenomeHeader);
        size_t chrom_size = n_chromosomes * sizeof(Chromosome);
        size_t gene_table_size = n_genes * sizeof(GeneDescriptor);
        size_t gene_data_offset = header_size + chrom_size + gene_table_size;
        
        // Align to cache line
        gene_data_offset = (gene_data_offset + CACHE_LINE - 1) & ~(CACHE_LINE - 1);
        
        // Calculate total gene data size
        size_t total_gene_data = 0;
        for (auto& w : gene_weights) {
            size_t s = w.size() * sizeof(float);
            s = (s + CACHE_LINE - 1) & ~(CACHE_LINE - 1); // align each gene
            total_gene_data += s;
        }
        
        // Epigenetic state
        size_t epi_offset = gene_data_offset + total_gene_data;
        epi_offset = (epi_offset + CACHE_LINE - 1) & ~(CACHE_LINE - 1);
        size_t epi_size = n_genes * sizeof(float) * 2; // methylation + history
        
        // Regulation network (sparse, placeholder for now)
        size_t reg_offset = epi_offset + epi_size;
        reg_offset = (reg_offset + CACHE_LINE - 1) & ~(CACHE_LINE - 1);
        
        size_t total_size = reg_offset + CACHE_LINE; // minimum
        
        // Create file
        int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) return false;
        
        // Extend file to total size
        ftruncate(fd, total_size);
        
        // mmap it
        void* base = mmap(nullptr, total_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { ::close(fd); return false; }
        
        memset(base, 0, total_size);
        
        // Write header
        GenomeHeader* hdr = reinterpret_cast<GenomeHeader*>(base);
        hdr->magic = GENOME_MAGIC;
        hdr->version = GENOME_VERSION;
        hdr->n_chromosomes = n_chromosomes;
        hdr->n_genes = n_genes;
        hdr->d_model = d_model;
        hdr->vocab_size = vocab_size;
        hdr->genome_length = total_size;
        hdr->gene_data_offset = gene_data_offset;
        hdr->epigenetic_offset = epi_offset;
        hdr->regulation_offset = reg_offset;
        
        // Write chromosomes
        Chromosome* c = reinterpret_cast<Chromosome*>(
            static_cast<char*>(base) + header_size);
        for (uint32_t i = 0; i < n_chromosomes; i++) {
            c[i] = chroms[i];
        }
        
        // Write gene descriptors + data
        GeneDescriptor* g = reinterpret_cast<GeneDescriptor*>(
            static_cast<char*>(base) + header_size + chrom_size);
        
        size_t data_cursor = gene_data_offset;
        for (uint32_t i = 0; i < n_genes; i++) {
            g[i] = gene_descs[i];
            g[i].data_offset = data_cursor;
            g[i].data_size = gene_weights[i].size() * sizeof(float);
            g[i].active = 1; // all genes start active
            
            // Write gene weight data
            memcpy(static_cast<char*>(base) + data_cursor,
                   gene_weights[i].data(),
                   gene_weights[i].size() * sizeof(float));
            
            // Align next gene
            data_cursor += g[i].data_size;
            data_cursor = (data_cursor + CACHE_LINE - 1) & ~(CACHE_LINE - 1);
        }
        
        // Initialize epigenetic state (all genes fully accessible)
        float* epi = reinterpret_cast<float*>(
            static_cast<char*>(base) + epi_offset);
        for (uint32_t i = 0; i < n_genes; i++) {
            epi[i] = 1.0f;                    // methylation = 1.0 (fully open)
            epi[n_genes + i] = 0.0f;          // expression history = 0
        }
        
        // Sync and close
        msync(base, total_size, MS_SYNC);
        munmap(base, total_size);
        ::close(fd);
        
        return true;
    }
    
    // ---- Nanosecond Gene Access ----
    
    /** Get raw pointer to gene weight data. O(1), nanoseconds. */
    inline float* gene_data(uint32_t gene_id) {
        return reinterpret_cast<float*>(
            static_cast<char*>(base_) + genes_[gene_id].data_offset);
    }
    
    /** Get gene descriptor. O(1). */
    inline const GeneDescriptor& gene(uint32_t gene_id) const {
        return genes_[gene_id];
    }
    
    /** Get chromosome. O(1). */
    inline const Chromosome& chromosome(uint32_t chrom_id) const {
        return chroms_[chrom_id];
    }
    
    /** Check if gene is expressed (epigenetic gating). */
    inline bool is_expressed(uint32_t gene_id) const {
        if (!genes_[gene_id].active) return false;
        if (!epi_methylation_) return true;
        return epi_methylation_[gene_id] > 0.5f;
    }
    
    /** Get expression level (continuous, for soft gating). */
    inline float expression_level(uint32_t gene_id) const {
        if (!epi_methylation_) return 1.0f;
        return epi_methylation_[gene_id];
    }
    
    // ---- Epigenetic Modification (the organism LEARNS) ----
    
    /** Methylate a gene (reduce accessibility). Persists to disk via mmap. */
    inline void methylate(uint32_t gene_id, float delta) {
        if (!epi_methylation_) return;
        epi_methylation_[gene_id] = std::max(0.0f, 
            epi_methylation_[gene_id] - delta);
    }
    
    /** Demethylate a gene (increase accessibility). */
    inline void demethylate(uint32_t gene_id, float delta) {
        if (!epi_methylation_) return;
        epi_methylation_[gene_id] = std::min(1.0f,
            epi_methylation_[gene_id] + delta);
    }
    
    /** Record that a gene was expressed (update history). */
    inline void record_expression(uint32_t gene_id) {
        if (!epi_history_) return;
        // Exponential moving average of expression frequency
        epi_history_[gene_id] = 0.99f * epi_history_[gene_id] + 0.01f;
    }
    
    /** Record that a gene was NOT expressed. */
    inline void record_silence(uint32_t gene_id) {
        if (!epi_history_) return;
        epi_history_[gene_id] = 0.99f * epi_history_[gene_id];
    }
    
    // ---- Transcription (Context → Gene Selection) ----
    
    /**
     * Transcribe: given input context, determine which genes to express.
     * Returns expression weights for all genes on a chromosome.
     */
    void transcribe(uint32_t chrom_id, const float* context, int d,
                    float* expression_weights) {
        const Chromosome& chr = chroms_[chrom_id];
        
        for (uint32_t i = 0; i < chr.gene_count; i++) {
            uint32_t gid = chr.gene_start + i;
            if (!genes_[gid].active) {
                expression_weights[i] = 0.0f;
                continue;
            }
            
            // Dot product between context and gene (affinity)
            float* gdata = gene_data(gid);
            float score = 0.0f;
            int dim = std::min(d, (int)genes_[gid].d_in);
            for (int j = 0; j < dim; j++) {
                score += context[j] * gdata[j];
            }
            score /= std::sqrt((float)dim);
            
            // Apply epigenetic gating
            score *= expression_level(gid);
            
            expression_weights[i] = score;
        }
        
        // Softmax over expression weights
        float max_score = -1e30f;
        for (uint32_t i = 0; i < chr.gene_count; i++) {
            if (expression_weights[i] > max_score) 
                max_score = expression_weights[i];
        }
        float sum = 0.0f;
        for (uint32_t i = 0; i < chr.gene_count; i++) {
            expression_weights[i] = std::exp(expression_weights[i] - max_score);
            sum += expression_weights[i];
        }
        for (uint32_t i = 0; i < chr.gene_count; i++) {
            expression_weights[i] /= sum;
        }
    }
    
    // ---- Info ----
    
    uint32_t n_genes() const { return header_->n_genes; }
    uint32_t n_chromosomes() const { return header_->n_chromosomes; }
    uint32_t d_model() const { return header_->d_model; }
    uint32_t vocab_size() const { return header_->vocab_size; }
    size_t file_size() const { return file_size_; }
};
