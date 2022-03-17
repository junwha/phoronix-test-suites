/*
* Copyright(c) 2019 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef EbIntraPrediction_AVX2_h
#define EbIntraPrediction_AVX2_h

#include "EbDefinitions.h"
#ifdef __cplusplus
extern "C" {
#endif
// Weights are quadratic from '1' to '1 / BlockSize', scaled by
// 2^sm_weight_log2_scale.
static const int32_t sm_weight_log2_scale = 8;
// max(block_size_wide[BLOCK_LARGEST], block_size_high[BLOCK_LARGEST])
#define MAX_BLOCK_DIM 64
/* clang-format off */
    static const uint8_t sm_weight_arrays[2 * MAX_BLOCK_DIM] = {
        // Unused, because we always offset by bs, which is at least 2.
        0, 0,
        // bs = 2
        255, 128,
        // bs = 4
        255, 149, 85, 64,
        // bs = 8
        255, 197, 146, 105, 73, 50, 37, 32,
        // bs = 16
        255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
        // bs = 32
        255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74,
        66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
        // bs = 64
        255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156,
        150, 144, 138, 133, 127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69,
        65, 61, 57, 54, 50, 47, 44, 41, 38, 35, 32, 29, 27, 25, 22, 20, 18, 16, 15,
        13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
    };

#ifdef __cplusplus
}
#endif
#endif // EbIntraPrediction_AVX2_h
