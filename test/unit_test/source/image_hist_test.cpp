#include "gtest_lite.hpp"
#include "image_hist.hpp"
#include "image_hist_rvv.hpp"
#include <cstdlib>

TEST(Histogram256, OutputMatchesScalar) {
    const size_t n = 10000;
    uint8_t input[n];
    uint32_t hist_ref[256];
    uint32_t hist_vec_opt[256];

    for(size_t i = 0; i < n; ++i) {
        input[i] = rand() % 256;
    }

    ref::hist256(input, n, hist_ref);
    vec::image_hist_rvv_optimized(input, n, hist_vec_opt);

    uint32_t total_ref = 0;
    uint32_t total_vec_opt = 0;

    for(int i = 0; i < 256; ++i) {
        ASSERT_EQ(hist_ref[i], hist_vec_opt[i]);
        
        total_ref += hist_ref[i];
        total_vec_opt += hist_vec_opt[i];
    }
    
    // Check that sum(hist[b]) equals the number of input elements
    ASSERT_EQ(total_ref, n);
    ASSERT_EQ(total_vec_opt, n);
}