/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbMemory_AVX2_h
#define EbMemory_AVX2_h

#include "EbDefinitions.h"
#include "vpx_dsp_rtcd.h"

#ifndef _mm256_set_m128i
#define _mm256_set_m128i(/* __m128i */ hi, /* __m128i */ lo) \
    _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)
#endif

#ifndef _mm256_setr_m128i
#define _mm256_setr_m128i(/* __m128i */ lo, /* __m128i */ hi) \
    _mm256_set_m128i((hi), (lo))
#endif

#define pair_set_epi16_avx2(a, b) \
  _mm256_set1_epi32((int)((uint16_t)(a) | ((uint32_t)(b) << 16) ))

static INLINE __m256i load8bit_4x4_avx2(const uint8_t *const src,
    const uint32_t stride)
{
    __m128i src01, src23;
    src01 = _mm_cvtsi32_si128(*(int32_t*)(src + 0 * stride));
    src01 = _mm_insert_epi32(src01, *(int32_t *)(src + 1 * stride), 1);
    src23 = _mm_cvtsi32_si128(*(int32_t*)(src + 2 * stride));
    src23 = _mm_insert_epi32(src23, *(int32_t *)(src + 3 * stride), 1);
    return _mm256_setr_m128i(src01, src23);
}

static INLINE __m256i load8bit_8x4_avx2(const uint8_t *const src,
    const uint32_t stride)
{
    __m128i src01, src23;
    src01 = _mm_loadl_epi64((__m128i *)(src + 0 * stride));
    src01 = _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(src01),
        (double *)(src + 1 * stride)));
    src23 = _mm_loadl_epi64((__m128i *)(src + 2 * stride));
    src23 = _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(src23),
        (double *)(src + 3 * stride)));
    return _mm256_setr_m128i(src01, src23);
}

static INLINE __m256i load8bit_16x2_avx2(const uint8_t *const src,
    const uint32_t stride)
{
    const __m128i src0 = _mm_load_si128((__m128i *)(src + 0 * stride));
    const __m128i src1 = _mm_load_si128((__m128i *)(src + 1 * stride));
    return _mm256_setr_m128i(src0, src1);
}

static INLINE __m256i load8bit_16x2_unaligned_avx2(const uint8_t *const src,
    const uint32_t stride)
{
    const __m128i src0 = _mm_loadu_si128((__m128i *)(src + 0 * stride));
    const __m128i src1 = _mm_loadu_si128((__m128i *)(src + 1 * stride));
    return _mm256_setr_m128i(src0, src1);
}

static INLINE __m256i load16bit_signed_8x2_avx2(const int16_t *const src,
    const uint32_t stride)
{
    const __m128i src0 = _mm_load_si128((__m128i *)(src + 0 * stride));
    const __m128i src1 = _mm_load_si128((__m128i *)(src + 1 * stride));
    return _mm256_setr_m128i(src0, src1);
}

static INLINE __m256i load16bit_signed_8x2_rev_avx2(const int16_t *const src,
    const uint32_t stride)
{
    const __m128i src0 = _mm_load_si128((__m128i *)(src + 0 * stride));
    const __m128i src1 = _mm_load_si128((__m128i *)(src + 1 * stride));
    return _mm256_set_m128i(src0, src1);
}

static INLINE void store16bit_signed_8x2_avx2(const __m256i output, const int16_t *const dst,
    const uint32_t stride)
{
    const __m128i output0 = _mm256_extracti128_si256(output, 0);
    const __m128i output1 = _mm256_extracti128_si256(output, 1);
    _mm_store_si128((__m128i *)(dst + 0 * stride), output0);
    _mm_store_si128((__m128i *)(dst + 1 * stride), output1);
}

#endif // EbMemory_AVX2_h
