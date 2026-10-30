#include "image_hist.hpp"

namespace ref 
{
    void hist256(const uint8_t* src, size_t n, uint32_t* hist)
    {
        for (int i = 0; i < 256; ++i) {
            hist[i] = 0;
        }

        for (size_t i = 0; i < n; ++i) {
            hist[src[i]]++;
        }
    }
}