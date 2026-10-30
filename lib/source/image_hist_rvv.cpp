#include "image_hist_rvv.hpp"
#include <riscv_vector.h>

namespace vec 
{
    // Optimized Vector Implementation (Hierarchical Structure)
    void image_hist_rvv_optimized(const uint8_t* src, size_t n, uint32_t* hist)
    {
        const size_t max_vl = __riscv_vsetvlmax_e32m8();
        uint32_t local_tables[max_vl * 256];

        size_t total = max_vl * 256;
        uint32_t* p = local_tables;
        while (total > 0) {
            size_t vl = __riscv_vsetvl_e32m8(total);
            vuint32m8_t zero = __riscv_vmv_v_x_u32m8(0, vl);
            __riscv_vse32_v_u32m8(p, zero, vl);
            p += vl;
            total -= vl;
        }

        // Hierarchical Optimization: Precomputing Base Offsets outside the loop
        vuint32m8_t v_lane = __riscv_vid_v_u32m8(max_vl);
        vuint32m8_t v_base_off = __riscv_vmul_vx_u32m8(v_lane, 1024, max_vl);

        while (n > 0) {
            size_t vl = __riscv_vsetvl_e8m2(n);

            vuint8m2_t  v_src  = __riscv_vle8_v_u8m2(src, vl);
            vuint32m8_t v_idx  = __riscv_vzext_vf4_u32m8(v_src, vl);
            vuint32m8_t v_idx_bytes = __riscv_vsll_vx_u32m8(v_idx, 2, vl);
            
            vuint32m8_t v_off = __riscv_vadd_vv_u32m8(v_base_off, v_idx_bytes, vl);

            vuint32m8_t v_cnt = __riscv_vluxei32_v_u32m8(local_tables, v_off, vl);
            v_cnt = __riscv_vadd_vx_u32m8(v_cnt, 1, vl);
            __riscv_vsuxei32_v_u32m8(local_tables, v_off, v_cnt, vl);

            src += vl;
            n   -= vl;
        }

        for (int bin = 0; bin < 256; ++bin) {
            uint32_t sum = 0;
            for (size_t lane = 0; lane < max_vl; ++lane) {
                sum += local_tables[lane * 256 + bin];
            }
            hist[bin] = sum;
        }
    }
}