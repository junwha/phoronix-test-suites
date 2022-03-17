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
#include "EbDefinitions.h"
#include "EbMcp_SSE2.h" // THIS SHOULD BE _SSE2 in the future
#include <emmintrin.h>
#include "common_dsp_rtcd.h"

void svt_picture_average_kernel_sse2_intrin(EbByte src0, uint32_t src0_stride, EbByte src1,
                                            uint32_t src1_stride, EbByte dst, uint32_t dst_stride,
                                            uint32_t area_width, uint32_t area_height) {
    __m128i  xmm_avg1, xmm_avg2;
    uint32_t y;
    assert((area_width & 3) == 0);
    assert((area_height & 1) == 0);

    if (area_width >= 16) {
        for (uint32_t x = 0; x < area_height; ++x) {
            for (y = 0; y + 15 < area_width; y += 16) {
                xmm_avg1 = _mm_avg_epu8(_mm_loadu_si128((__m128i *)(src0 + y)),
                                        _mm_loadu_si128((__m128i *)(src1 + y)));
                _mm_storeu_si128((__m128i *)(dst + y), xmm_avg1);
            }
            if (area_width & 8) {
                xmm_avg1 = _mm_avg_epu8(_mm_loadl_epi64((__m128i *)(src0 + y)),
                                        _mm_loadl_epi64((__m128i *)(src1 + y)));
                _mm_storel_epi64((__m128i *)(dst + y), xmm_avg1);
                y += 8;
            }
            if (area_width & 4) {
                xmm_avg1               = _mm_avg_epu8(_mm_cvtsi32_si128(*(uint32_t *)(src0 + y)),
                                        _mm_cvtsi32_si128(*(uint32_t *)(src1 + y)));
                *(uint32_t *)(dst + y) = _mm_cvtsi128_si32(xmm_avg1);
            }

            src0 += src0_stride;
            src1 += src1_stride;
            dst += dst_stride;
        }
    } else if (area_width == 4) {
        for (y = 0; y < area_height; y += 2) {
            xmm_avg1 = _mm_avg_epu8(_mm_cvtsi32_si128(*(uint32_t *)src0),
                                    _mm_cvtsi32_si128(*(uint32_t *)src1));
            xmm_avg2 = _mm_avg_epu8(_mm_cvtsi32_si128(*(uint32_t *)(src0 + src0_stride)),
                                    _mm_cvtsi32_si128(*(uint32_t *)(src1 + src1_stride)));

            *(uint32_t *)dst                = _mm_cvtsi128_si32(xmm_avg1);
            *(uint32_t *)(dst + dst_stride) = _mm_cvtsi128_si32(xmm_avg2);

            src0 += src0_stride << 1;
            src1 += src1_stride << 1;
            dst += dst_stride << 1;
        }
    } else if (area_width == 8) {
        for (y = 0; y < area_height; y += 2) {
            xmm_avg1 = _mm_avg_epu8(_mm_loadl_epi64((__m128i *)src0),
                                    _mm_loadl_epi64((__m128i *)src1));
            xmm_avg2 = _mm_avg_epu8(_mm_loadl_epi64((__m128i *)(src0 + src0_stride)),
                                    _mm_loadl_epi64((__m128i *)(src1 + src1_stride)));

            _mm_storel_epi64((__m128i *)dst, xmm_avg1);
            _mm_storel_epi64((__m128i *)(dst + dst_stride), xmm_avg2);

            src0 += src0_stride << 1;
            src1 += src1_stride << 1;
            dst += dst_stride << 1;
        }
    }
}

void svt_picture_average_kernel1_line_sse2_intrin(EbByte src0, EbByte src1, EbByte dst,
                                                  uint32_t area_width) {
    __m128i xmm_avg1, xmm_avg2, xmm_avg3, xmm_avg4;

    if (area_width > 16) {
        if (area_width == 32) {
            //for (y = 0; y < area_height; y += 2)
            {
                xmm_avg1 = _mm_avg_epu8(_mm_loadu_si128((__m128i *)src0),
                                        _mm_loadu_si128((__m128i *)src1));
                xmm_avg2 = _mm_avg_epu8(_mm_loadu_si128((__m128i *)(src0 + 16)),
                                        _mm_loadu_si128((__m128i *)(src1 + 16)));
                //xmm_avg3 = _mm_avg_epu8(_mm_loadu_si128((__m128i*)(src0 + src0_stride)), _mm_loadu_si128((__m128i*)(src1 + src1_stride)));
                //xmm_avg4 = _mm_avg_epu8(_mm_loadu_si128((__m128i*)(src0 + src0_stride + 16)), _mm_loadu_si128((__m128i*)(src1 + src1_stride + 16)));

                _mm_storeu_si128((__m128i *)dst, xmm_avg1);
                _mm_storeu_si128((__m128i *)(dst + 16), xmm_avg2);
                //_mm_storeu_si128((__m128i*) (dst + dst_stride), xmm_avg3);
                //_mm_storeu_si128((__m128i*) (dst + dst_stride + 16), xmm_avg4);

                //src0 += src0_stride << 1;
                //src1 += src1_stride << 1;
                //dst += dst_stride << 1;
            }
        } else {
            //for (y = 0; y < area_height; y += 2)
            {
                xmm_avg1 = _mm_avg_epu8(_mm_loadu_si128((__m128i *)src0),
                                        _mm_loadu_si128((__m128i *)src1));
                xmm_avg2 = _mm_avg_epu8(_mm_loadu_si128((__m128i *)(src0 + 16)),
                                        _mm_loadu_si128((__m128i *)(src1 + 16)));
                xmm_avg3 = _mm_avg_epu8(_mm_loadu_si128((__m128i *)(src0 + 32)),
                                        _mm_loadu_si128((__m128i *)(src1 + 32)));
                xmm_avg4 = _mm_avg_epu8(_mm_loadu_si128((__m128i *)(src0 + 48)),
                                        _mm_loadu_si128((__m128i *)(src1 + 48)));

                //xmm_avg5 = _mm_avg_epu8(_mm_loadu_si128((__m128i*)(src0 + src0_stride)), _mm_loadu_si128((__m128i*)(src1 + src1_stride)));
                //xmm_avg6 = _mm_avg_epu8(_mm_loadu_si128((__m128i*)(src0 + src0_stride + 16)), _mm_loadu_si128((__m128i*)(src1 + src1_stride + 16)));
                //xmm_avg7 = _mm_avg_epu8(_mm_loadu_si128((__m128i*)(src0 + src0_stride + 32)), _mm_loadu_si128((__m128i*)(src1 + src1_stride + 32)));
                //xmm_avg8 = _mm_avg_epu8(_mm_loadu_si128((__m128i*)(src0 + src0_stride + 48)), _mm_loadu_si128((__m128i*)(src1 + src1_stride + 48)));

                _mm_storeu_si128((__m128i *)dst, xmm_avg1);
                _mm_storeu_si128((__m128i *)(dst + 16), xmm_avg2);
                _mm_storeu_si128((__m128i *)(dst + 32), xmm_avg3);
                _mm_storeu_si128((__m128i *)(dst + 48), xmm_avg4);

                //_mm_storeu_si128((__m128i*) (dst + dst_stride), xmm_avg5);
                //_mm_storeu_si128((__m128i*) (dst + dst_stride + 16), xmm_avg6);
                //_mm_storeu_si128((__m128i*) (dst + dst_stride + 32), xmm_avg7);
                //_mm_storeu_si128((__m128i*) (dst + dst_stride + 48), xmm_avg8);

                //src0 += src0_stride << 1;
                //src1 += src1_stride << 1;
                //dst += dst_stride << 1;
            }
        }
    } else {
        if (area_width == 16) {
            //for (y = 0; y < area_height; y += 2)
            {
                xmm_avg1 = _mm_avg_epu8(_mm_loadu_si128((__m128i *)src0),
                                        _mm_loadu_si128((__m128i *)src1));
                //xmm_avg2 = _mm_avg_epu8(_mm_loadu_si128((__m128i*)(src0 + src0_stride)), _mm_loadu_si128((__m128i*)(src1 + src1_stride)));

                _mm_storeu_si128((__m128i *)dst, xmm_avg1);
                //_mm_storeu_si128((__m128i*) (dst + dst_stride), xmm_avg2);

                //src0 += src0_stride << 1;
                //src1 += src1_stride << 1;
                //dst += dst_stride << 1;
            }
        } else if (area_width == 4) {
            //for (y = 0; y < area_height; y += 2)
            {
                xmm_avg1 = _mm_avg_epu8(_mm_cvtsi32_si128(*(uint32_t *)src0),
                                        _mm_cvtsi32_si128(*(uint32_t *)src1));
                //xmm_avg2 = _mm_avg_epu8(_mm_cvtsi32_si128(*(uint32_t *)(src0 + src0_stride)), _mm_cvtsi32_si128(*(uint32_t *)(src1 + src1_stride)));

                *(uint32_t *)dst = _mm_cvtsi128_si32(xmm_avg1);
                //*(uint32_t *)(dst + dst_stride) = _mm_cvtsi128_si32(xmm_avg2);

                //src0 += src0_stride << 1;
                //src1 += src1_stride << 1;
                //dst += dst_stride << 1;
            }
        } else if (area_width == 8) {
            //for (y = 0; y < area_height; y += 2)
            {
                xmm_avg1 = _mm_avg_epu8(_mm_loadl_epi64((__m128i *)src0),
                                        _mm_loadl_epi64((__m128i *)src1));
                //xmm_avg2 = _mm_avg_epu8(_mm_loadl_epi64((__m128i*)(src0 + src0_stride)), _mm_loadl_epi64((__m128i*)(src1 + src1_stride)));

                _mm_storel_epi64((__m128i *)dst, xmm_avg1);
                //_mm_storel_epi64((__m128i*) (dst + dst_stride), xmm_avg2);

                //src0 += src0_stride << 1;
                //src1 += src1_stride << 1;
                //dst += dst_stride << 1;
            }
        } else {
            //for (y = 0; y < area_height; y += 2)
            {
                xmm_avg1 = _mm_avg_epu8(_mm_loadl_epi64((__m128i *)src0),
                                        _mm_loadl_epi64((__m128i *)src1));
                xmm_avg2 = _mm_avg_epu8(_mm_cvtsi32_si128(*(uint32_t *)(src0 + 8)),
                                        _mm_cvtsi32_si128(*(uint32_t *)(src1 + 8)));

                //xmm_avg3 = _mm_avg_epu8(_mm_loadl_epi64((__m128i*)(src0 + src0_stride)), _mm_loadl_epi64((__m128i*)(src1 + src1_stride)));
                //xmm_avg4 = _mm_avg_epu8(_mm_cvtsi32_si128(*(uint32_t *)(src0 + src0_stride + 8)), _mm_cvtsi32_si128(*(uint32_t *)(src1 + src1_stride + 8)));

                _mm_storel_epi64((__m128i *)dst, xmm_avg1);
                *(uint32_t *)(dst + 8) = _mm_cvtsi128_si32(xmm_avg2);
                //_mm_storel_epi64((__m128i*) (dst + dst_stride), xmm_avg3);
                //*(uint32_t *)(dst + dst_stride + 8) = _mm_cvtsi128_si32(xmm_avg4);

                //src0 += src0_stride << 1;
                //src1 += src1_stride << 1;
                //dst += dst_stride << 1;
            }
        }
    }
}
