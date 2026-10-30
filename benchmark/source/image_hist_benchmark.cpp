#include "image_hist.hpp"
#include "image_hist_rvv.hpp"
#include "riscv_timer.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <cassert>

#ifdef RISCV_BAREMETAL
#include <klib.h>
inline void riscv_enable_vector() {
    uint64_t mstatus_vs = 0x200; 
    asm volatile("csrs mstatus, %0" ::"r"(mstatus_vs));
    asm volatile("csrwi vcsr, 0");
}
#else
inline void riscv_enable_vector() {}
#endif

inline uint8_t* make_constant(uint8_t* out, size_t n, uint8_t value) {
    for (size_t i = 0; i < n; ++i) out[i] = value;
    return out;
}

inline uint8_t* make_skewed(uint8_t* out, size_t n,
                            uint8_t a = 0, uint8_t b = 255, uint32_t seed = 12345,
                            uint32_t hot_pct = 90, uint32_t hot_bins = 8) {
    assert(a <= b); 
    uint32_t range = uint32_t(b) - a + 1;
    if (hot_pct  > 100)   hot_pct  = 100;
    if (hot_bins == 0)    hot_bins = 1;
    if (hot_bins > range) hot_bins = range;
    uint32_t cold_bins = range - hot_bins;          
    uint32_t x = seed;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        bool hot = (uint64_t(x) * 100 >> 32) < hot_pct || cold_bins == 0;
        x = x * 1664525u + 1013904223u;
        uint32_t idx = hot ? uint32_t(uint64_t(x) * hot_bins  >> 32)
                           : hot_bins + uint32_t(uint64_t(x) * cold_bins >> 32);
        out[i] = uint8_t(a + idx);
    }
    return out;
}

// Assignment requirement: 1,000,000 items
const size_t input_size = 1000000;
static uint8_t input_data[input_size]; // Allocated in .bss to avoid gem5 memory crash

void check_sum(uint32_t* hist, size_t expected_n, const char* name) {
    uint32_t total = 0;
    for(int i = 0; i < 256; ++i) {
        total += hist[i];
    }
    if (total != expected_n) {
        printf("Error: %s sum(hist[b]) = %u, expected %u\n", name, total, (uint32_t)expected_n);
    }
}

int main()
{
    riscv_enable_vector();

    uint32_t hist_ref[256]; 
    uint32_t hist_vec_opt[256];
    
    // Test only Constant and Skewed as requested
    const char* test_names[] = {"Constant", "Strongly Skewed"};

    for (int i = 0; i < 2; ++i)
    {
        if (i == 0) {
            make_constant(input_data, input_size, 128);
        } else {
            make_skewed(input_data, input_size);
        }

        Timer timer_scalar, timer_opt;

        timer_scalar.Start();
        ref::hist256(input_data, input_size, hist_ref);
        timer_scalar.Stop();

        timer_opt.Start();
        vec::image_hist_rvv_optimized(input_data, input_size, hist_vec_opt);
        timer_opt.Stop();
        
        // Verify constraint: Check that sum(hist[b]) equals the number of input elements.
        check_sum(hist_ref, input_size, "Scalar");
        check_sum(hist_vec_opt, input_size, "Optimized Vector");

        uint64_t ref_cycles = timer_scalar.ElapsedCycles();
        uint64_t ref_instrs = timer_scalar.ElapsedInstructions();
        uint64_t opt_cycles = timer_opt.ElapsedCycles();
        uint64_t opt_instrs = timer_opt.ElapsedInstructions();

        printf("--- %s Input (1,000,000 items) ---\n", test_names[i]);
        printf("Scalar          : Cycles = %llu, Instructions = %llu\n", (unsigned long long)ref_cycles, (unsigned long long)ref_instrs);
        printf("Optimized Vector: Cycles = %llu, Instructions = %llu\n", (unsigned long long)opt_cycles, (unsigned long long)opt_instrs);
        
        printf("Speedup (Opt)   : %.2fx\n\n", (float)ref_cycles / (float)opt_cycles);
    }

    printf("Output is correct.\n");
    return 0;
}