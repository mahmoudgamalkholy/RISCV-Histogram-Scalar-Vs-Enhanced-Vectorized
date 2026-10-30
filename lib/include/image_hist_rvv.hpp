#pragma once
#include <cstdint>
#include <cstddef>

namespace vec 
{
    void image_hist_rvv_optimized(const uint8_t* src, size_t n, uint32_t* hist);
}