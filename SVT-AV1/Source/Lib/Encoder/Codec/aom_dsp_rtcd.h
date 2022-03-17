// clang-format off
/*
* Copyright(c) 2019 Intel Corporation
* Copyright (c) 2016, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

// This file is generated. Do not edit.
#ifndef AOM_DSP_RTCD_H_
#define AOM_DSP_RTCD_H_

#include "common_dsp_rtcd.h"
#include "EbDefinitions.h"
#include "EbCodingUnit.h"

#undef RTCD_EXTERN
#ifdef AOM_RTCD_C
#define RTCD_EXTERN                //CHKN RTCD call in effect. declare the function pointers in  encHandle.
#else
#define RTCD_EXTERN extern         //CHKN run time externing the fucntion pointers.
#endif

#ifdef __GNUC__
#define LIKELY(v) __builtin_expect(v, 1)
#define UNLIKELY(v) __builtin_expect(v, 0)
#else
#define LIKELY(v) (v)
#define UNLIKELY(v) (v)
#endif

 /**************************************
 * Instruction Set Support
 **************************************/

#ifdef _WIN32
# include <intrin.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

//    // Helper Functions
//    CPU_FLAGS get_cpu_flags();
//    CPU_FLAGS get_cpu_flags_to_use();
    void setup_rtcd_internal(CPU_FLAGS flags);

    //to not include convolve.h, just forward declare what's needed.
    struct ConvolveParams;
    struct InterpFilterParams;

    int64_t svt_aom_sse_c(const uint8_t *a, int a_stride, const uint8_t *b, int b_stride, int width, int height);
    RTCD_EXTERN int64_t(*svt_aom_sse)(const uint8_t *a, int a_stride, const uint8_t *b, int b_stride, int width, int height);
    int64_t svt_aom_highbd_sse_c(const uint8_t *a8, int a_stride, const uint8_t *b8, int b_stride, int width, int height);
    RTCD_EXTERN int64_t(*svt_aom_highbd_sse)(const uint8_t *a8, int a_stride, const uint8_t *b8, int b_stride, int width, int height);
    void svt_av1_wedge_compute_delta_squares_c(int16_t *d, const int16_t *a, const int16_t *b, int N);
    RTCD_EXTERN void(*svt_av1_wedge_compute_delta_squares)(int16_t *d, const int16_t *a, const int16_t *b, int N);
    int8_t svt_av1_wedge_sign_from_residuals_c(const int16_t *ds, const uint8_t *m, int N, int64_t limit);
    RTCD_EXTERN int8_t(*svt_av1_wedge_sign_from_residuals)(const int16_t *ds, const uint8_t *m, int N, int64_t limit);
    uint64_t compute_cdef_dist_c(const uint16_t *dst, int32_t dstride, const uint16_t *src, const CdefList *dlist, int32_t cdef_count, BlockSize bsize, int32_t coeff_shift, int32_t pli, uint8_t subsampling_factor);
    RTCD_EXTERN uint64_t(*svt_compute_cdef_dist_16bit)(const uint16_t *dst, int32_t dstride, const uint16_t *src, const CdefList *dlist, int32_t cdef_count, BlockSize bsize, int32_t coeff_shift, int32_t pli, uint8_t subsampling_factor);
    uint64_t compute_cdef_dist_8bit_c(const uint8_t *dst8, int32_t dstride, const uint8_t *src8, const CdefList *dlist, int32_t cdef_count, BlockSize bsize, int32_t coeff_shift, int32_t pli, uint8_t subsampling_factor);
    RTCD_EXTERN uint64_t(*svt_compute_cdef_dist_8bit)(const uint8_t *dst8, int32_t dstride, const uint8_t *src8, const CdefList *dlist, int32_t cdef_count, BlockSize bsize, int32_t coeff_shift, int32_t pli, uint8_t subsampling_factor);
    void svt_av1_compute_stats_c(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H);
    RTCD_EXTERN void(*svt_av1_compute_stats)(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H);
    void svt_av1_compute_stats_highbd_c(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H, AomBitDepth bit_depth);
    RTCD_EXTERN void(*svt_av1_compute_stats_highbd)(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H, AomBitDepth bit_depth);
    typedef uint64_t(*EbSpatialFullDistType)(
        uint8_t   *input,
        uint32_t   input_offset,
        uint32_t   input_stride,
        uint8_t   *recon,
        int32_t    recon_offset,
        uint32_t   recon_stride,
        uint32_t   area_width,
        uint32_t   area_height);
    int64_t svt_av1_lowbd_pixel_proj_error_c(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride, const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride, int32_t *flt1, int32_t flt1_stride, int32_t xq[2], const SgrParamsType *params);
    RTCD_EXTERN int64_t(*svt_av1_lowbd_pixel_proj_error)(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride, const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride, int32_t *flt1, int32_t flt1_stride, int32_t xq[2], const SgrParamsType *params);
    int64_t svt_av1_highbd_pixel_proj_error_c(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride, const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride, int32_t *flt1, int32_t flt1_stride, int32_t xq[2], const SgrParamsType *params);
    RTCD_EXTERN int64_t(*svt_av1_highbd_pixel_proj_error)(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride, const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride, int32_t *flt1, int32_t flt1_stride, int32_t xq[2], const SgrParamsType *params);
    void svt_subtract_average_c(int16_t *pred_buf_q3, int32_t width, int32_t height, int32_t round_offset, int32_t num_pel_log2);
    RTCD_EXTERN void(*svt_subtract_average)(int16_t *pred_buf_q3, int32_t width, int32_t height, int32_t round_offset, int32_t num_pel_log2);
    int64_t svt_av1_calc_frame_error_c(const uint8_t *const ref, int stride, const uint8_t *const dst, int p_width, int p_height, int p_stride);
    RTCD_EXTERN int64_t(*svt_av1_calc_frame_error)(const uint8_t *const ref, int stride, const uint8_t *const dst, int p_width, int p_height, int p_stride);
    void svt_av1_fwd_txfm2d_4x16_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_4x16)(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x4)(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x8_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_4x8)(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x4)(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x16_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x16)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x8_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x8)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x16_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_4x16)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x8_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_4x8)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x16_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x16)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x8_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x8)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x32_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x32)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x32_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x32)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x64_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x64)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x32_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_64x32)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x64_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x64)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x16_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_64x16)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_transform_two_d_64x64_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_64x64)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_transform_two_d_32x32_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x32)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_transform_two_d_16x16_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x16)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_transform_two_d_8x8_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x8)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_transform_two_d_4x4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_4x4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x16_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x16_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x8_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x8_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x16_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_4x16_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x4_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x4_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x8_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_4x8_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x4_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x4_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x16_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x16_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x8_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x8_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x32_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x32_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x32_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x32_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x64_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x64_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x32_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_64x32_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x64_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x64_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x16_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_64x16_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_transform_two_d_64x64_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_64x64_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_transform_two_d_32x32_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x32_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_transform_two_d_16x16_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x16_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_transform_two_d_8x8_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x8_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_transform_two_d_4x4_N2_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_4x4_N2)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x16_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x16_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x8_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x8_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x16_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_4x16_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x4_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x4_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x8_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_4x8_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x4_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x4_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x16_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x16_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x8_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x8_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x32_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x32_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x32_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x32_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x64_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x64_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x32_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_64x32_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x64_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x64_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x16_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_64x16_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_transform_two_d_64x64_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_64x64_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_transform_two_d_32x32_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_32x32_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_transform_two_d_16x16_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_16x16_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_transform_two_d_8x8_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_8x8_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_transform_two_d_4x4_N4_c(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    RTCD_EXTERN void(*svt_av1_fwd_txfm2d_4x4_N4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    int svt_aom_satd_c(const TranLow *coeff, int length);
    RTCD_EXTERN int(*svt_aom_satd)(const TranLow *coeff, int length);
    int64_t svt_av1_block_error_c(const TranLow *coeff, const TranLow *dqcoeff, intptr_t block_size, int64_t *ssz);
    RTCD_EXTERN int64_t(*svt_av1_block_error)(const TranLow *coeff, const TranLow *dqcoeff, intptr_t block_size, int64_t *ssz);
    RTCD_EXTERN void(*svt_smooth_v_predictor)(uint8_t *dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint8_t *above, const uint8_t *left);
    void svt_get_proj_subspace_c(const uint8_t *src8, int width, int height, int src_stride, const uint8_t *dat8, int dat_stride, int use_highbitdepth, int32_t *flt0, int flt0_stride, int32_t *flt1, int flt1_stride, int *xq, const SgrParamsType *params);
    RTCD_EXTERN void(*svt_get_proj_subspace)(const uint8_t *src8, int width, int height, int src_stride, const uint8_t *dat8, int dat_stride, int use_highbitdepth, int32_t *flt0, int flt0_stride, int32_t *flt1, int flt1_stride, int *xq, const SgrParamsType *params);
    uint64_t svt_handle_transform16x64_c(int32_t *output);
    RTCD_EXTERN uint64_t(*svt_handle_transform16x64)(int32_t *output);
    uint64_t svt_handle_transform32x64_c(int32_t *output);
    RTCD_EXTERN uint64_t(*svt_handle_transform32x64)(int32_t *output);
    uint64_t svt_handle_transform64x16_c(int32_t *output);
    RTCD_EXTERN uint64_t(*svt_handle_transform64x16)(int32_t *output);
    uint64_t svt_handle_transform64x32_c(int32_t *output);
    RTCD_EXTERN uint64_t(*svt_handle_transform64x32)(int32_t *output);
    uint64_t svt_handle_transform64x64_c(int32_t *output);
    RTCD_EXTERN uint64_t(*svt_handle_transform64x64)(int32_t *output);
    uint64_t svt_handle_transform16x64_N2_N4_c(int32_t *output);
    uint64_t svt_handle_transform16x64_N2_N4_avx2(int32_t *output);
    RTCD_EXTERN uint64_t(*svt_handle_transform16x64_N2_N4)(int32_t *output);
    uint64_t svt_handle_transform32x64_N2_N4_c(int32_t *output);
    uint64_t svt_handle_transform32x64_N2_N4_avx2(int32_t *output);
    RTCD_EXTERN uint64_t(*svt_handle_transform32x64_N2_N4)(int32_t *output);
    uint64_t svt_handle_transform64x16_N2_N4_c(int32_t *output);
    uint64_t svt_handle_transform64x16_N2_N4_avx2(int32_t *output);
    RTCD_EXTERN uint64_t(*svt_handle_transform64x16_N2_N4)(int32_t *output);
    uint64_t svt_handle_transform64x32_N2_N4_c(int32_t *output);
    uint64_t svt_handle_transform64x32_N2_N4_avx2(int32_t *output);
    RTCD_EXTERN uint64_t(*svt_handle_transform64x32_N2_N4)(int32_t *output);
    uint64_t svt_handle_transform64x64_N2_N4_c(int32_t *output);
    uint64_t svt_handle_transform64x64_N2_N4_avx2(int32_t *output);
    RTCD_EXTERN uint64_t(*svt_handle_transform64x64_N2_N4)(int32_t *output);
    uint64_t svt_search_one_dual_c(int *lev0, int *lev1, int nb_strengths, uint64_t** mse[2], int sb_count, int start_gi, int end_gi);
    RTCD_EXTERN uint64_t(*svt_search_one_dual)(int *lev0, int *lev1, int nb_strengths, uint64_t** mse[2], int sb_count, int start_gi, int end_gi);
    uint32_t svt_aom_mse16x16_c(const uint8_t *src_ptr, int32_t  source_stride, const uint8_t *ref_ptr, int32_t  recon_stride, uint32_t *sse);
    RTCD_EXTERN uint32_t(*svt_aom_mse16x16)(const uint8_t *src_ptr, int32_t  source_stride, const uint8_t *ref_ptr, int32_t  recon_stride, uint32_t *sse);
    void svt_aom_quantize_b_c(const TranLow *coeff_ptr, int32_t stride,int32_t width, int32_t height, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr,const int16_t *quant_ptr, const int16_t *quant_shift_ptr,TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr,uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan,const QmVal *qm_ptr, const QmVal *iqm_ptr, const int32_t log_scale);
    void svt_aom_quantize_b_c_ii(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, const QmVal *qm_ptr, const QmVal *iqm_ptr, const int32_t log_scale);
    RTCD_EXTERN void(*svt_aom_quantize_b)(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr,const int16_t *quant_ptr, const int16_t *quant_shift_ptr,TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr,uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan,const QmVal *qm_ptr, const QmVal *iqm_ptr, const int32_t log_scale);
    void svt_aom_highbd_quantize_b_c(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, const QmVal *qm_ptr, const QmVal *iqm_ptr, const int32_t log_scale);
    RTCD_EXTERN void(*svt_aom_highbd_quantize_b)(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, const QmVal *qm_ptr, const QmVal *iqm_ptr, const int32_t log_scale);
    void svt_av1_quantize_fp_c(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
    RTCD_EXTERN void(*svt_av1_quantize_fp)(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
    void svt_av1_highbd_quantize_fp_c(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, int16_t log_scale);
    RTCD_EXTERN void(*svt_av1_highbd_quantize_fp)(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, int16_t log_scale);
    void svt_av1_quantize_fp_32x32_c(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
    RTCD_EXTERN void(*svt_av1_quantize_fp_32x32)(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
    void svt_av1_quantize_fp_64x64_c(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
    RTCD_EXTERN void(*svt_av1_quantize_fp_64x64)(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
    void svt_aom_highbd_8_mse16x16_c(const uint8_t *src_ptr, int32_t  source_stride, const uint8_t *ref_ptr, int32_t  recon_stride, uint32_t *sse);
    RTCD_EXTERN void(*svt_aom_highbd_8_mse16x16)(const uint8_t *src_ptr, int32_t  source_stride, const uint8_t *ref_ptr, int32_t  recon_stride, uint32_t *sse);
    uint32_t svt_aom_sad128x128_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad128x128)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad128x128x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad128x128x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad128x64_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad128x64)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad128x64x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad128x64x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad16x16_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad16x16)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad16x16x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad16x16x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad16x32_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad16x32)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad16x32x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad16x32x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad16x4_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad16x4)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad16x4x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad16x4x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad16x64_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad16x64)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad16x64x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad16x64x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad16x8_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad16x8)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad16x8x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad16x8x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad32x16_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad32x16)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad32x16x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad32x16x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad32x32_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad32x32)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad32x32x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad32x32x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad32x64_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad32x64)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad32x64x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad32x64x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad32x8_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad32x8)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad32x8x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad32x8x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad4x16_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad4x16)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad4x16x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad4x16x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad4x4_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad4x4)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad4x4x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad4x4x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad4x8_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad4x8)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad4x8x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad4x8x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad64x128_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad64x128)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad64x128x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad64x128x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad64x16_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad64x16)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad64x16x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad64x16x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad64x32_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad64x32)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad64x32x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad64x32x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad64x64_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad64x64)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad64x64x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad64x64x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad8x16_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad8x16)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad8x16x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad8x16x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad8x32_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad8x32)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad8x32x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad8x32x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad8x4_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad8x4)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad8x4x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad8x4x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    uint32_t svt_aom_sad8x8_c(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    RTCD_EXTERN uint32_t(*svt_aom_sad8x8)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    void svt_aom_sad8x8x4d_c(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    RTCD_EXTERN void(*svt_aom_sad8x8x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
    void svt_aom_upsampled_pred_c(MacroBlockD *xd, const struct AV1Common *const cm, int mi_row, int mi_col, const MV *const mv, uint8_t *comp_pred, int width, int height, int subpel_x_q3, int subpel_y_q3, const uint8_t *ref, int ref_stride, int subpel_search);
    RTCD_EXTERN void(*svt_aom_upsampled_pred) (MacroBlockD *xd, const struct AV1Common *const cm, int mi_row, int mi_col, const MV *const mv, uint8_t *comp_pred, int width, int height, int subpel_x_q3, int subpel_y_q3, const uint8_t *ref, int ref_stride, int subpel_search);
    unsigned int svt_aom_obmc_sad128x128_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad128x128)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad128x64_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad128x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad16x16_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad16x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad16x32_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad16x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad16x4_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad16x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad16x64_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad16x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad16x8_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad16x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad32x16_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad32x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad32x32_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad32x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad32x64_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad32x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad32x8_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad32x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad4x16_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad4x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad4x4_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad4x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad4x8_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad4x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad64x128_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad64x128)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad64x16_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad64x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad64x32_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad64x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad64x64_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad64x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad8x16_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad8x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad8x32_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad8x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad8x4_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad8x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sad8x8_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sad8x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
    unsigned int svt_aom_obmc_sub_pixel_variance128x128_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance128x128)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance128x64_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance128x64)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance16x16_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance16x16)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance16x32_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance16x32)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance16x4_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance16x4)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance16x64_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance16x64)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance16x8_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance16x8)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance32x16_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance32x16)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance32x32_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance32x32)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance32x64_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance32x64)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance32x8_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance32x8)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance4x16_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance4x16)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance4x4_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance4x4)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance4x8_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance4x8)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance64x128_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance64x128)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance64x16_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance64x16)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance64x32_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance64x32)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance64x64_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance64x64)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance8x16_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance8x16)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance8x32_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance8x32)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance8x4_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance8x4)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_sub_pixel_variance8x8_c(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_sub_pixel_variance8x8)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance128x128_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance128x128)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance128x64_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance128x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance16x16_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance16x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance16x32_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance16x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance16x4_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance16x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance16x64_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance16x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance16x8_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance16x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance32x16_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance32x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance32x32_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance32x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance32x64_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance32x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance32x8_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance32x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance4x16_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance4x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance4x4_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance4x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance4x8_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance4x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance64x128_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance64x128)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance64x16_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance64x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance64x32_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance64x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance64x64_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance64x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance8x16_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance8x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance8x32_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance8x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance8x4_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance8x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_obmc_variance8x8_c(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_obmc_variance8x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
    unsigned int svt_aom_variance4x4_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance4x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance4x8_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance4x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance4x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance4x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance8x4_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance8x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance8x8_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance8x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance8x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance8x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance8x32_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance8x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance16x4_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance16x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance16x8_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance16x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance16x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance16x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance16x32_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance16x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance16x64_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance16x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance32x8_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance32x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance32x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance32x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance32x32_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance32x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance32x64_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance32x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance64x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance64x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance64x32_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance64x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance64x64_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance64x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance64x128_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance64x128)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance128x64_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance128x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance128x128_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_variance128x128)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance4x4_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance4x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance4x8_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance4x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance4x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance4x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance8x4_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance8x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance8x8_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance8x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance8x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance8x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance8x32_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance8x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x4_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance16x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x8_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance16x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance16x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x32_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance16x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x64_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance16x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x8_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance32x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance32x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x32_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance32x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x64_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance32x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance64x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x32_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance64x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x64_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance64x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x128_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance64x128)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance128x64_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance128x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance128x128_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    RTCD_EXTERN unsigned int(*svt_aom_highbd_10_variance128x128)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    uint32_t svt_aom_sub_pixel_variance128x128_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance128x128_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance128x128_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance128x128_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance128x128_avx512(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance128x128)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance128x64_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance128x64_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance128x64_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance128x64_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance128x64_avx512(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance128x64)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance16x16_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance16x16_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance16x16_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance16x16_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance16x16)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance16x32_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance16x32_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance16x32_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance16x32_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance16x32)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance16x4_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance16x4_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance16x4_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance16x4_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance16x4)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance16x64_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance16x64_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance16x64_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance16x64_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance16x64)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance16x8_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance16x8_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance16x8_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance16x8_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance16x8)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance32x16_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance32x16_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance32x16_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance32x16_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance32x16_avx512(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance32x16)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance32x32_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance32x32_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance32x32_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance32x32_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance32x32_avx512(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance32x32)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance32x64_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance32x64_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance32x64_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance32x64_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance32x64_avx512(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance32x64)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance32x8_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance32x8_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance32x8_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance32x8)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance4x16_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance4x16_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance4x16_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance4x16)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance4x4_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance4x4_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance4x4_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance4x4)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance4x8_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance4x8_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance4x8_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance4x8)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance64x128_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance64x128_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance64x128_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance64x128_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance64x128_avx512(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance64x128)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance64x16_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance64x16_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance64x16_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance64x16)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance64x32_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance64x32_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance64x32_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance64x32_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance64x32_avx512(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance64x32)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance64x64_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance64x64_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance64x64_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance64x64_avx2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance64x64_avx512(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance64x64)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance8x16_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance8x16_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance8x16_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance8x16)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance8x32_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance8x32_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance8x32_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance8x32)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance8x4_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance8x4_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance8x4_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance8x4)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);

    uint32_t svt_aom_sub_pixel_variance8x8_c(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#ifdef ARCH_X86_64
    uint32_t svt_aom_sub_pixel_variance8x8_sse2(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    uint32_t svt_aom_sub_pixel_variance8x8_ssse3(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
#endif
    RTCD_EXTERN uint32_t(*svt_aom_sub_pixel_variance8x8)(const uint8_t *src_ptr, int source_stride, int xoffset, int  yoffset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse);
    void svt_aom_ifft16x16_float_c(const float *input, float *temp, float *output);
    RTCD_EXTERN void(*svt_aom_ifft16x16_float)(const float *input, float *temp, float *output);
    void svt_aom_ifft2x2_float_c(const float *input, float *temp, float *output);
    RTCD_EXTERN void(*svt_aom_ifft2x2_float)(const float *input, float *temp, float *output);
    void svt_aom_ifft32x32_float_c(const float *input, float *temp, float *output);
    RTCD_EXTERN void(*svt_aom_ifft32x32_float)(const float *input, float *temp, float *output);
    void svt_aom_ifft4x4_float_c(const float *input, float *temp, float *output);
    RTCD_EXTERN void(*svt_aom_ifft4x4_float)(const float *input, float *temp, float *output);
    void svt_aom_ifft8x8_float_c(const float *input, float *temp, float *output);
    RTCD_EXTERN void(*svt_aom_ifft8x8_float)(const float *input, float *temp, float *output);
    void svt_aom_fft16x16_float_c(const float *input, float *temp, float *output);
    RTCD_EXTERN void(*svt_aom_fft16x16_float)(const float *input, float *temp, float *output);
    void svt_aom_fft2x2_float_c(const float *input, float *temp, float *output);
    RTCD_EXTERN void(*svt_aom_fft2x2_float)(const float *input, float *temp, float *output);
    void svt_aom_fft32x32_float_c(const float *input, float *temp, float *output);
    RTCD_EXTERN void(*svt_aom_fft32x32_float)(const float *input, float *temp, float *output);
    void svt_aom_fft4x4_float_c(const float *input, float *temp, float *output);
    RTCD_EXTERN void(*svt_aom_fft4x4_float)(const float *input, float *temp, float *output);
    void svt_aom_fft8x8_float_c(const float *input, float *temp, float *output);
    RTCD_EXTERN void(*svt_aom_fft8x8_float)(const float *input, float *temp, float *output);
    void svt_av1_get_nz_map_contexts_c(const uint8_t *const levels, const int16_t *const scan, const uint16_t eob, const TxSize tx_size, const TxClass tx_class, int8_t *const coeff_contexts);
    RTCD_EXTERN void(*svt_av1_get_nz_map_contexts)(const uint8_t *const levels, const int16_t *const scan, const uint16_t eob, const TxSize tx_size, const TxClass tx_class, int8_t *const coeff_contexts);
    RTCD_EXTERN void(*svt_sad_loop_kernel)(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint64_t *best_sad, int16_t *x_search_center, int16_t *y_search_center, uint32_t src_stride_raw, uint8_t skip_search_line, int16_t search_area_width, int16_t search_area_height);
    void svt_av1_txb_init_levels_c(const TranLow *const coeff, const int32_t width, const int32_t height, uint8_t *const levels);
    RTCD_EXTERN void(*svt_av1_txb_init_levels)(const TranLow *const coeff, const int32_t width, const int32_t height, uint8_t *const levels);
    void svt_av1_get_gradient_hist_c(const uint8_t *src, int src_stride, int rows, int cols, uint64_t *hist);
    RTCD_EXTERN void(*svt_av1_get_gradient_hist)(const uint8_t *src, int src_stride, int rows, int cols, uint64_t *hist);
    double svt_av1_compute_cross_correlation_c(unsigned char *im1, int stride1, int x1, int y1, unsigned char *im2, int stride2, int x2, int y2);
    RTCD_EXTERN double(*svt_av1_compute_cross_correlation)(unsigned char *im1, int stride1, int x1, int y1, unsigned char *im2, int stride2, int x2, int y2);
    void svt_av1_k_means_dim1_c(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr);
    RTCD_EXTERN void(*svt_av1_k_means_dim1)(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr);
    void svt_av1_k_means_dim2_c(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr);
    RTCD_EXTERN void(*svt_av1_k_means_dim2)(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr);
    void svt_av1_calc_indices_dim1_c(const int* data, const int* centroids, uint8_t* indices, int n, int k);
    RTCD_EXTERN void(*svt_av1_calc_indices_dim1)(const int* data, const int* centroids, uint8_t* indices, int n, int k);
    void svt_av1_calc_indices_dim2_c(const int* data, const int* centroids, uint8_t* indices, int n, int k);
    RTCD_EXTERN void(*svt_av1_calc_indices_dim2)(const int* data, const int* centroids, uint8_t* indices, int n, int k);
    RTCD_EXTERN void(*svt_av1_apply_filtering)(const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre, int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src, int uv_src_stride, const uint8_t *u_pre, const uint8_t *v_pre, int uv_pre_stride, unsigned int block_width, unsigned int block_height, int ss_x, int ss_y, int strength, const int *blk_fw, int use_whole_blk, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum, uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count);
    RTCD_EXTERN void(*svt_av1_apply_filtering_highbd)(const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre, int y_pre_stride, const uint16_t *u_src, const uint16_t *v_src, int uv_src_stride, const uint16_t *u_pre, const uint16_t *v_pre, int uv_pre_stride, unsigned int block_width, unsigned int block_height, int ss_x, int ss_y, int strength, const int *blk_fw, int use_whole_blk, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum, uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count);

    struct MeContext;

    RTCD_EXTERN void(*svt_av1_apply_temporal_filter_planewise_fast)(
        struct MeContext *context_ptr, const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre,
        int y_pre_stride, unsigned int block_width,
        unsigned int block_height, uint32_t *y_accum, uint16_t *y_count);

    RTCD_EXTERN void(*svt_av1_apply_temporal_filter_planewise_fast_hbd)(
        struct MeContext *context_ptr, const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre,
        int y_pre_stride, unsigned int block_width,
        unsigned int block_height, uint32_t *y_accum, uint16_t *y_count, uint32_t encoder_bit_depth);

    RTCD_EXTERN void(*svt_av1_apply_temporal_filter_planewise_medium)(
        struct MeContext *context_ptr, const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre,
        int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src, int uv_src_stride,
        const uint8_t *u_pre, const uint8_t *v_pre, int uv_pre_stride, unsigned int block_width,
        unsigned int block_height, int ss_x, int ss_y, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
        uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count);

    RTCD_EXTERN void(*svt_av1_apply_temporal_filter_planewise_medium_hbd)(
        struct MeContext *context_ptr, const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre,
        int y_pre_stride, const uint16_t *u_src, const uint16_t *v_src, int uv_src_stride,
        const uint16_t *u_pre, const uint16_t *v_pre, int uv_pre_stride, unsigned int block_width,
        unsigned int block_height, int ss_x, int ss_y, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
        uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count, uint32_t encoder_bit_depth);

    RTCD_EXTERN void(*svt_av1_apply_temporal_filter_planewise)(
        struct MeContext *context_ptr, const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre, int y_pre_stride,
        const uint8_t *u_src, const uint8_t *v_src, int uv_src_stride, const uint8_t *u_pre,
        const uint8_t *v_pre, int uv_pre_stride, unsigned int block_width, unsigned int block_height,
        int ss_x, int ss_y,
        uint32_t *y_accum,
        uint16_t *y_count, uint32_t *u_accum, uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count);

    RTCD_EXTERN void(*svt_av1_apply_temporal_filter_planewise_hbd)(
        struct MeContext *context_ptr, const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre,
        int y_pre_stride, const uint16_t *u_src, const uint16_t *v_src, int uv_src_stride,
        const uint16_t *u_pre, const uint16_t *v_pre, int uv_pre_stride, unsigned int block_width,
        unsigned int block_height, int ss_x, int ss_y,
        uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
        uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count, uint32_t encoder_bit_depth);
    struct MeContext;
    void get_final_filtered_pixels_c(struct MeContext *context_ptr, EbByte *src_center_ptr_start, uint16_t **altref_buffer_highbd_start, uint32_t **accum, uint16_t **count, const uint32_t *stride, int blk_y_src_offset, int blk_ch_src_offset, uint16_t blk_width_ch, uint16_t blk_height_ch, Bool is_highbd);
    void get_final_filtered_pixels_sse4_1(struct MeContext *context_ptr, EbByte *src_center_ptr_start, uint16_t **altref_buffer_highbd_start, uint32_t **accum, uint16_t **count, const uint32_t *stride, int blk_y_src_offset, int blk_ch_src_offset, uint16_t blk_width_ch, uint16_t blk_height_ch, Bool is_highbd);
    void get_final_filtered_pixels_avx2(struct MeContext *context_ptr, EbByte *src_center_ptr_start, uint16_t **altref_buffer_highbd_start, uint32_t **accum, uint16_t **count, const uint32_t *stride, int blk_y_src_offset, int blk_ch_src_offset, uint16_t blk_width_ch, uint16_t blk_height_ch, Bool is_highbd);
    RTCD_EXTERN void (*get_final_filtered_pixels)(struct MeContext *context_ptr, EbByte *src_center_ptr_start, uint16_t **altref_buffer_highbd_start, uint32_t **accum, uint16_t **count, const uint32_t *stride, int blk_y_src_offset, int blk_ch_src_offset, uint16_t blk_width_ch, uint16_t blk_height_ch, Bool is_highbd);
    void apply_filtering_central_sse4_1(struct MeContext *context_ptr, EbPictureBufferDesc *input_picture_ptr_central, EbByte *src, uint32_t **accum, uint16_t **count, uint16_t blk_width, uint16_t blk_height, uint32_t ss_x, uint32_t ss_y);
    void apply_filtering_central_avx2(struct MeContext *context_ptr, EbPictureBufferDesc *input_picture_ptr_central, EbByte *src, uint32_t **accum, uint16_t **count, uint16_t blk_width, uint16_t blk_height, uint32_t ss_x, uint32_t ss_y);
    void apply_filtering_central_c(struct MeContext *context_ptr, EbPictureBufferDesc *input_picture_ptr_central, EbByte *src, uint32_t **accum, uint16_t **count, uint16_t blk_width, uint16_t blk_height, uint32_t ss_x, uint32_t ss_y);
    RTCD_EXTERN void (*apply_filtering_central)(struct MeContext *context_ptr, EbPictureBufferDesc *input_picture_ptr_central, EbByte *src, uint32_t **accum, uint16_t **count, uint16_t blk_width, uint16_t blk_height, uint32_t ss_x, uint32_t ss_y);
    void apply_filtering_central_highbd_sse4_1(struct MeContext *context_ptr, EbPictureBufferDesc *input_picture_ptr_central, uint16_t **src_16bit, uint32_t **accum, uint16_t **count, uint16_t blk_width, uint16_t blk_height, uint32_t ss_x, uint32_t ss_y);
    void apply_filtering_central_highbd_avx2(struct MeContext *context_ptr, EbPictureBufferDesc *input_picture_ptr_central, uint16_t **src_16bit, uint32_t **accum, uint16_t **count, uint16_t blk_width, uint16_t blk_height, uint32_t ss_x, uint32_t ss_y);
    void apply_filtering_central_highbd_c(struct MeContext *context_ptr, EbPictureBufferDesc *input_picture_ptr_central, uint16_t **src_16bit, uint32_t **accum, uint16_t **count, uint16_t blk_width, uint16_t blk_height, uint32_t ss_x, uint32_t ss_y);
    RTCD_EXTERN void (*apply_filtering_central_highbd)(struct MeContext *context_ptr, EbPictureBufferDesc *input_picture_ptr_central, uint16_t **src_16bit, uint32_t **accum, uint16_t **count, uint16_t blk_width, uint16_t blk_height, uint32_t ss_x, uint32_t ss_y);
    void downsample_2d_sse4_1(uint8_t *input_samples, uint32_t input_stride, uint32_t input_area_width, uint32_t input_area_height, uint8_t *decim_samples, uint32_t decim_stride, uint32_t decim_step);
    void downsample_2d_avx2(uint8_t *input_samples, uint32_t input_stride, uint32_t input_area_width, uint32_t input_area_height, uint8_t *decim_samples, uint32_t decim_stride, uint32_t decim_step);
    RTCD_EXTERN void (*downsample_2d)(uint8_t *input_samples, uint32_t input_stride, uint32_t input_area_width, uint32_t input_area_height, uint8_t *decim_samples, uint32_t decim_stride, uint32_t decim_step);
    RTCD_EXTERN void(*svt_ext_sad_calculation_8x8_16x16)(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t *p_best_sad_8x8, uint32_t *p_best_sad_16x16, uint32_t *p_best_mv8x8, uint32_t *p_best_mv16x16, uint32_t mv, uint32_t *p_sad16x16, uint32_t *p_sad8x8, Bool sub_sad);
    void svt_ext_sad_calculation_8x8_16x16_c(uint8_t *src, uint32_t src_stride, uint8_t *ref,
        uint32_t ref_stride, uint32_t *p_best_sad_8x8,
        uint32_t *p_best_sad_16x16, uint32_t *p_best_mv8x8,
        uint32_t *p_best_mv16x16, uint32_t mv, uint32_t *p_sad16x16,
        uint32_t *p_sad8x8, Bool sub_sad);
    RTCD_EXTERN void(*svt_ext_sad_calculation_32x32_64x64)(uint32_t *p_sad16x16, uint32_t *p_best_sad_32x32, uint32_t *p_best_sad_64x64, uint32_t *p_best_mv32x32, uint32_t *p_best_mv64x64, uint32_t mv, uint32_t *p_sad32x32);
    void svt_ext_sad_calculation_32x32_64x64_c(uint32_t *p_sad16x16, uint32_t *p_best_sad_32x32,
        uint32_t *p_best_sad_64x64, uint32_t *p_best_mv32x32,
        uint32_t *p_best_mv64x64, uint32_t mv, uint32_t *p_sad32x32);

    RTCD_EXTERN void(*svt_ext_all_sad_calculation_8x8_16x16)(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t mv, uint8_t out_8x8, uint32_t *p_best_sad_8x8, uint32_t *p_best_sad_16x16, uint32_t *p_best_mv8x8, uint32_t *p_best_mv16x16, uint32_t p_eight_sad16x16[16][8], uint32_t p_eight_sad8x8[64][8], Bool sub_sad);
    RTCD_EXTERN void(*svt_ext_eight_sad_calculation_32x32_64x64)(uint32_t p_sad16x16[16][8], uint32_t *p_best_sad_32x32, uint32_t *p_best_sad_64x64, uint32_t *p_best_mv32x32, uint32_t *p_best_mv64x64, uint32_t mv, uint32_t p_sad32x32[4][8]);
    RTCD_EXTERN void(*svt_initialize_buffer_32bits)(uint32_t* pointer, uint32_t count128, uint32_t count32, uint32_t value);
    RTCD_EXTERN uint32_t(*svt_nxm_sad_kernel_sub_sampled)(const uint8_t *src, uint32_t src_stride, const uint8_t *ref, uint32_t ref_stride, uint32_t height, uint32_t width);
    RTCD_EXTERN uint32_t(*svt_nxm_sad_kernel)(const uint8_t *src, uint32_t src_stride, const uint8_t *ref, uint32_t ref_stride, uint32_t height, uint32_t width);
    RTCD_EXTERN uint32_t(*nxm_sad_avg_kernel)(uint8_t *src, uint32_t src_stride, uint8_t *ref1, uint32_t ref1_stride, uint8_t *ref2, uint32_t ref2_stride, uint32_t height, uint32_t width);
    RTCD_EXTERN uint64_t(*svt_compute_mean_8x8)(uint8_t *input_samples, uint32_t input_stride, uint32_t input_area_width, uint32_t input_area_height);
    RTCD_EXTERN uint64_t(*svt_compute_mean_square_values_8x8)(uint8_t *input_samples, uint32_t input_stride, uint32_t input_area_width, uint32_t input_area_height);
    RTCD_EXTERN uint64_t(*svt_compute_sub_mean_8x8)(uint8_t* input_samples, uint16_t input_stride);
    uint64_t svt_compute_sub_mean_8x8_c(uint8_t* input_samples, uint16_t input_stride);
    RTCD_EXTERN void(*svt_compute_interm_var_four8x8)(uint8_t *input_samples, uint16_t input_stride, uint64_t *mean_of8x8_blocks, uint64_t *mean_of_squared8x8_blocks);
    RTCD_EXTERN uint32_t(*sad_16b_kernel)(uint16_t *src, uint32_t src_stride, uint16_t *ref, uint32_t ref_stride, uint32_t height, uint32_t width);
    struct svt_mv_cost_param;
    void svt_pme_sad_loop_kernel_c(const struct svt_mv_cost_param *mv_cost_params, uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint32_t *best_cost, int16_t *best_mvx, int16_t *best_mvy, int16_t search_position_start_x, int16_t search_position_start_y, int16_t search_area_width, int16_t search_area_height, int16_t search_step, int16_t mvx, int16_t mvy);
    void svt_pme_sad_loop_kernel_sse4_1(const struct svt_mv_cost_param *mv_cost_params, uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint32_t *best_cost, int16_t *best_mvx, int16_t *best_mvy, int16_t search_position_start_x, int16_t search_position_start_y, int16_t search_area_width, int16_t search_area_height, int16_t search_step, int16_t mvx, int16_t mvy);
    void svt_pme_sad_loop_kernel_avx2(const struct svt_mv_cost_param *mv_cost_params, uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint32_t *best_cost, int16_t *best_mvx, int16_t *best_mvy, int16_t search_position_start_x, int16_t search_position_start_y, int16_t search_area_width, int16_t search_area_height, int16_t search_step, int16_t mvx, int16_t mvy);
    RTCD_EXTERN void(*svt_pme_sad_loop_kernel)(const struct svt_mv_cost_param *mv_cost_params, uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint32_t *best_cost, int16_t *best_mvx, int16_t *best_mvy, int16_t search_position_start_x, int16_t search_position_start_y, int16_t search_area_width, int16_t search_area_height, int16_t search_step, int16_t mvx, int16_t mvy);
    RTCD_EXTERN uint32_t(*variance_highbd)(const uint16_t *a, int a_stride, const uint16_t *b, int b_stride, int w, int h, uint32_t *sse);
    uint32_t variance_highbd_c(const uint16_t *a, int a_stride, const uint16_t *b, int b_stride, int w, int h, uint32_t *sse);
    RTCD_EXTERN int(*svt_av1_haar_ac_sad_8x8_uint8_input)(uint8_t *input, int stride, int hbd);
    int svt_av1_haar_ac_sad_8x8_uint8_input_c(uint8_t *input, int stride, int hbd);
    void svt_unpack_and_2bcompress_sse4_1(uint16_t *in16b_buffer, uint32_t in16b_stride, uint8_t *out8b_buffer, uint32_t out8b_stride, uint8_t *out2b_buffer, uint32_t out2b_stride, uint32_t width, uint32_t height);
    void svt_unpack_and_2bcompress_avx2(uint16_t *in16b_buffer, uint32_t in16b_stride, uint8_t *out8b_buffer, uint32_t out8b_stride, uint8_t *out2b_buffer, uint32_t out2b_stride, uint32_t width, uint32_t height);
    RTCD_EXTERN void (*svt_unpack_and_2bcompress)(uint16_t *in16b_buffer, uint32_t in16b_stride, uint8_t *out8b_buffer, uint32_t out8b_stride,uint8_t *out2b_buffer, uint32_t out2b_stride, uint32_t width, uint32_t height);
#ifdef ARCH_X86_64
    int64_t svt_aom_sse_avx2(const uint8_t *a, int a_stride, const uint8_t *b, int b_stride, int width, int height);
    int64_t svt_aom_highbd_sse_avx2(const uint8_t *a8, int a_stride, const uint8_t *b8, int b_stride, int width, int height);

    void svt_av1_wedge_compute_delta_squares_avx2(int16_t *d, const int16_t *a, const int16_t *b, int N);


    int8_t svt_av1_wedge_sign_from_residuals_sse2(const int16_t *ds, const uint8_t *m, int N, int64_t limit);
    int8_t svt_av1_wedge_sign_from_residuals_avx2(const int16_t *ds, const uint8_t *m, int N, int64_t limit);
    uint64_t compute_cdef_dist_16bit_sse4_1(const uint16_t *dst, int32_t dstride, const uint16_t *src, const CdefList *dlist, int32_t cdef_count, BlockSize bsize, int32_t coeff_shift, int32_t pli, uint8_t subsampling_factor);
    uint64_t compute_cdef_dist_16bit_avx2(const uint16_t *dst, int32_t dstride, const uint16_t *src, const CdefList *dlist, int32_t cdef_count, BlockSize bsize, int32_t coeff_shift, int32_t pli, uint8_t subsampling_factor);
    uint64_t compute_cdef_dist_8bit_sse4_1(const uint8_t *dst8, int32_t dstride, const uint8_t *src8, const CdefList *dlist, int32_t cdef_count, BlockSize bsize, int32_t coeff_shift, int32_t pli, uint8_t subsampling_factor);
    uint64_t compute_cdef_dist_8bit_avx2(const uint8_t *dst8, int32_t dstride, const uint8_t *src8, const CdefList *dlist, int32_t cdef_count, BlockSize bsize, int32_t coeff_shift, int32_t pli, uint8_t subsampling_factor);
    void svt_av1_compute_stats_sse4_1(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H);
    void svt_av1_compute_stats_avx2(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H);
    void svt_av1_compute_stats_avx512(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H);

    void svt_av1_compute_stats_highbd_sse4_1(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H, AomBitDepth bit_depth);
    void svt_av1_compute_stats_highbd_avx2(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H, AomBitDepth bit_depth);
    void svt_av1_compute_stats_highbd_avx512(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H, AomBitDepth bit_depth);

    int64_t svt_av1_lowbd_pixel_proj_error_sse4_1(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride, const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride, int32_t *flt1, int32_t flt1_stride, int32_t xq[2], const SgrParamsType *params);
    int64_t svt_av1_lowbd_pixel_proj_error_avx2(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride, const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride, int32_t *flt1, int32_t flt1_stride, int32_t xq[2], const SgrParamsType *params);
    int64_t svt_av1_lowbd_pixel_proj_error_avx512(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride, const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride, int32_t *flt1, int32_t flt1_stride, int32_t xq[2], const SgrParamsType *params);

    int64_t svt_av1_highbd_pixel_proj_error_sse4_1(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride, const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride, int32_t *flt1, int32_t flt1_stride, int32_t xq[2], const SgrParamsType *params);
    int64_t svt_av1_highbd_pixel_proj_error_avx2(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride, const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride, int32_t *flt1, int32_t flt1_stride, int32_t xq[2], const SgrParamsType *params);

    void svt_subtract_average_avx2(int16_t *pred_buf_q3, int32_t width, int32_t height, int32_t round_offset, int32_t num_pel_log2);

    int64_t svt_av1_calc_frame_error_avx2(const uint8_t *const ref, int stride, const uint8_t *const dst, int p_width, int p_height, int p_stride);

    void svt_av1_fwd_txfm2d_4x16_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_16x4_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_4x8_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_8x4_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_8x16_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_16x8_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);





    void svt_av1_fwd_txfm2d_32x16_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_32x16_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_32x8_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_8x32_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_16x32_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_16x32_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_32x64_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_32x64_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_64x32_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_64x32_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_16x64_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_16x64_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_64x16_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_64x16_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_64x64_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_64x64_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_32x32_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_32x32_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_16x16_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_16x16_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_8x8_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_4x4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_4x8_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x16_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_8x4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x8_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x16_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x32_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_16x4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x8_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x16_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x32_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x64_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_32x8_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x16_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x32_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x64_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_64x16_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x32_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x64_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_av1_fwd_txfm2d_4x16_N2_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x4_N2_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x8_N2_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x4_N2_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x16_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x8_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x16_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x8_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x32_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x32_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x64_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x32_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x64_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x16_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x64_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x32_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x16_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x8_N2_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x4_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x16_N4_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x4_N4_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x8_N4_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x4_N4_avx2(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x16_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x8_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x16_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x8_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x32_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x32_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x64_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x32_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x64_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x16_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x64_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x32_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x16_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x8_N4_avx2(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x4_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_32x32_N2_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_32x64_N2_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_64x32_N2_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_64x64_N2_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void av1_fwd_txfm2d_64x64_N4_avx512(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x16_N2_sse4_1(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x4_N2_sse4_1(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x8_N2_sse4_1(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x4_N2_sse4_1(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x16_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x8_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x16_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x8_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x32_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x32_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x64_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x32_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x64_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x16_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x64_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x32_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x16_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x8_N2_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x16_N4_sse4_1(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x4_N4_sse4_1(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_4x8_N4_sse4_1(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x4_N4_sse4_1(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x16_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x8_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x16_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x8_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x32_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x32_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x64_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x32_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x64_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x16_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_64x64_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_32x32_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_16x16_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
    void svt_av1_fwd_txfm2d_8x8_N4_sse4_1(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);

    void svt_get_proj_subspace_avx2(const uint8_t *src8, int width, int height, int src_stride, const uint8_t *dat8, int dat_stride, int use_highbitdepth, int32_t *flt0, int flt0_stride, int32_t *flt1, int flt1_stride, int *xq, const SgrParamsType *params);

    uint64_t svt_handle_transform16x64_avx2(int32_t *output);

    uint64_t svt_handle_transform32x64_avx2(int32_t *output);

    uint64_t svt_handle_transform64x16_avx2(int32_t *output);

    uint64_t svt_handle_transform64x32_avx2(int32_t *output);

    uint64_t svt_handle_transform64x64_avx2(int32_t *output);
    uint64_t svt_search_one_dual_avx2(int *lev0, int *lev1, int nb_strengths, uint64_t** mse[2], int sb_count, int start_gi, int end_gi);
    uint64_t svt_search_one_dual_avx512(int *lev0, int *lev1, int nb_strengths, uint64_t** mse[2], int sb_count, int start_gi, int end_gi);
    uint32_t svt_aom_mse16x16_sse2(const uint8_t *src_ptr, int32_t  source_stride, const uint8_t *ref_ptr, int32_t  recon_stride, uint32_t *sse);
    uint32_t svt_aom_mse16x16_avx2(const uint8_t *src_ptr, int32_t  source_stride, const uint8_t *ref_ptr, int32_t  recon_stride, uint32_t *sse);

    void svt_aom_quantize_b_sse4_1(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, const QmVal *qm_ptr, const QmVal *iqm_ptr, const int32_t log_scale);
    void svt_aom_quantize_b_avx2(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, const QmVal *qm_ptr, const QmVal *iqm_ptr, const int32_t log_scale);

    void svt_aom_highbd_quantize_b_sse4_1(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, const QmVal *qm_ptr, const QmVal *iqm_ptr, const int32_t log_scale);
    void svt_aom_highbd_quantize_b_avx2(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, const QmVal *qm_ptr, const QmVal *iqm_ptr, const int32_t log_scale);

    void svt_av1_quantize_fp_sse4_1(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
    void svt_av1_quantize_fp_avx2(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);

    void svt_av1_highbd_quantize_fp_sse4_1(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, int16_t log_scale);
    void svt_av1_highbd_quantize_fp_avx2(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, int16_t log_scale);

    void svt_av1_quantize_fp_32x32_sse4_1(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
    void svt_av1_quantize_fp_32x32_avx2(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);

    void svt_av1_quantize_fp_64x64_sse4_1(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
    void svt_av1_quantize_fp_64x64_avx2(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);

    void svt_aom_highbd_8_mse16x16_sse2(const uint8_t *src_ptr, int32_t  source_stride, const uint8_t *ref_ptr, int32_t  recon_stride, uint32_t *sse);

    /* DC_PRED top */

    uint32_t svt_aom_sad128x128_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    uint32_t svt_aom_sad128x128_avx512(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad128x128x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);
    void svt_aom_sad128x128x4d_avx512(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);

    uint32_t svt_aom_sad128x64_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    uint32_t svt_aom_sad128x64_avx512(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad128x64x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);
    void svt_aom_sad128x64x4d_avx512(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad16x16_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad16x16x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad16x32_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad16x32x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad16x4_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad16x4x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad16x64_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad16x64x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad16x8_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad16x8x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad32x16_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad32x16x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad32x32_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad32x32x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad32x64_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad32x64x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad32x8_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad32x8x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad4x16_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad4x16x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad4x4_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad4x4x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad4x8_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad4x8x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad64x128_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    uint32_t svt_aom_sad64x128_avx512(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad64x128x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad64x16_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    uint32_t svt_aom_sad64x16_avx512(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad64x16x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad64x32_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    uint32_t svt_aom_sad64x32_avx512(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad64x32x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad64x64_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
    uint32_t svt_aom_sad64x64_avx512(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad64x64x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad8x16_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad8x16x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad8x32_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad8x32x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad8x4_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad8x4x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    uint32_t svt_aom_sad8x8_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);

    void svt_aom_sad8x8x4d_avx2(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[4], int ref_stride, uint32_t sad_array[4]);

    void svt_aom_upsampled_pred_sse2(MacroBlockD *xd, const struct AV1Common *const cm, int mi_row, int mi_col, const MV *const mv, uint8_t *comp_pred, int width, int height, int subpel_x_q3, int subpel_y_q3, const uint8_t *ref, int ref_stride, int subpel_search);

    unsigned int svt_aom_obmc_sad128x128_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad128x64_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad16x16_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad16x32_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad16x4_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad16x64_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad16x8_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad32x16_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad32x32_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad32x64_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad32x8_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad4x16_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad4x4_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad4x8_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad64x128_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad64x16_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad64x32_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad64x64_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad8x16_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad8x32_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad8x4_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sad8x8_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);

    unsigned int svt_aom_obmc_sub_pixel_variance128x128_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance128x64_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance16x16_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance16x32_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance16x4_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance16x64_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance16x8_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance32x16_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance32x32_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance32x64_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance32x8_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance4x16_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance4x4_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance4x8_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance64x128_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance64x16_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance64x32_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance64x64_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance8x16_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance8x32_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance8x4_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_sub_pixel_variance8x8_sse4_1(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance128x128_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance128x64_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance16x16_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance16x32_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance16x4_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance16x64_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance16x8_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance32x16_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance32x32_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance32x64_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance32x8_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance4x16_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance4x4_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance4x8_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance64x128_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance64x16_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance64x32_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance64x64_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance8x16_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance8x32_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance8x4_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance8x8_avx2(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance128x128_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance128x64_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance16x16_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance16x32_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance16x4_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance16x64_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance16x8_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance32x16_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance32x32_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance32x64_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance32x8_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance4x16_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance4x4_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance4x8_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance64x128_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance64x16_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance64x32_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance64x64_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance8x16_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance8x32_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance8x4_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);

    unsigned int svt_aom_obmc_variance8x8_sse4_1(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);


    unsigned int svt_aom_variance4x4_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance4x8_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance4x16_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance8x4_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance8x8_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance8x16_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance8x32_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance16x4_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance16x8_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance16x16_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance16x32_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance16x64_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance32x8_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance32x16_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance32x32_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance32x64_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance64x16_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance64x32_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance64x64_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance64x128_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance128x64_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance128x128_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance32x8_avx512(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance32x16_avx512(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance32x32_avx512(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance32x64_avx512(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance64x16_avx512(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance64x32_avx512(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance64x64_avx512(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance64x128_avx512(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance128x64_avx512(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_variance128x128_avx512(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance16x4_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance16x8_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance16x16_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance16x32_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance16x64_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance32x8_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance32x16_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance32x32_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance32x64_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance64x16_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance64x32_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance64x64_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance64x128_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance128x64_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_variance128x128_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_highbd_10_variance8x8_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance8x16_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance8x32_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x4_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x8_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x16_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x32_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x64_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x8_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x16_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x32_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x64_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x16_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x32_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x64_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x128_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance128x64_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance128x128_sse2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    unsigned int svt_aom_highbd_10_variance8x8_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance8x16_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance8x32_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x8_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x16_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x32_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance16x64_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x8_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x16_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x32_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance32x64_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x16_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x32_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x64_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance64x128_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance128x64_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
    unsigned int svt_aom_highbd_10_variance128x128_avx2(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

    void svt_aom_ifft16x16_float_avx2(const float *input, float *temp, float *output);


    void svt_aom_ifft32x32_float_avx2(const float *input, float *temp, float *output);

    void svt_aom_ifft4x4_float_sse2(const float *input, float *temp, float *output);

    void svt_aom_ifft8x8_float_avx2(const float *input, float *temp, float *output);

    void svt_aom_fft16x16_float_avx2(const float *input, float *temp, float *output);


    void svt_aom_fft32x32_float_avx2(const float *input, float *temp, float *output);

    void svt_aom_fft4x4_float_sse2(const float *input, float *temp, float *output);

    void svt_aom_fft8x8_float_avx2(const float *input, float *temp, float *output);

    void svt_av1_get_nz_map_contexts_sse2(const uint8_t *const levels, const int16_t *const scan, const uint16_t eob, const TxSize tx_size, const TxClass tx_class, int8_t *const coeff_contexts);
    void svt_av1_get_nz_map_contexts_avx2(const uint8_t *const levels, const int16_t *const scan, const uint16_t eob, const TxSize tx_size, const TxClass tx_class, int8_t *const coeff_contexts);

    void svt_residual_kernel8bit_avx2(uint8_t *input, uint32_t input_stride, uint8_t *pred, uint32_t pred_stride, int16_t *residual, uint32_t residual_stride, uint32_t area_width, uint32_t area_height);
    void svt_residual_kernel8bit_avx512(uint8_t *input, uint32_t input_stride, uint8_t *pred, uint32_t pred_stride, int16_t *residual, uint32_t residual_stride, uint32_t area_width, uint32_t area_height);

    void svt_sad_loop_kernel_sse4_1_intrin(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint64_t *best_sad, int16_t *x_search_center, int16_t *y_search_center, uint32_t src_stride_raw, uint8_t skip_search_line, int16_t search_area_width, int16_t search_area_height);
    void svt_sad_loop_kernel_avx2_intrin(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint64_t *best_sad, int16_t *x_search_center, int16_t *y_search_center, uint32_t src_stride_raw, uint8_t skip_search_line, int16_t search_area_width, int16_t search_area_height);
    void svt_sad_loop_kernel_avx512_intrin(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint64_t *best_sad, int16_t *x_search_center, int16_t *y_search_center, uint32_t src_stride_raw, uint8_t skip_search_line, int16_t search_area_width, int16_t search_area_height);

    void svt_av1_txb_init_levels_sse4_1(const TranLow *const coeff, const int32_t width, const int32_t height, uint8_t *const levels);
    void svt_av1_txb_init_levels_avx2(const TranLow *const coeff, const int32_t width, const int32_t height, uint8_t *const levels);
    void svt_av1_txb_init_levels_avx512(const TranLow *const coeff, const int32_t width, const int32_t height, uint8_t *const levels);
    int svt_aom_satd_avx2(const TranLow *coeff, int length);
    int64_t svt_av1_block_error_avx2(const TranLow *coeff, const TranLow *dqcoeff, intptr_t block_size, int64_t *ssz);
    void svt_av1_get_gradient_hist_avx2(const uint8_t *src, int src_stride, int rows, int cols, uint64_t *hist);

    double svt_av1_compute_cross_correlation_sse4_1(unsigned char *im1, int stride1, int x1, int y1, unsigned char *im2, int stride2, int x2, int y2);
    double svt_av1_compute_cross_correlation_avx2(unsigned char *im1, int stride1, int x1, int y1, unsigned char *im2, int stride2, int x2, int y2);

    void svt_av1_k_means_dim1_avx2(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr);

    void svt_av1_k_means_dim2_avx2(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr);

    void svt_av1_calc_indices_dim1_avx2(const int* data, const int* centroids, uint8_t* indices, int n, int k);

    void svt_av1_calc_indices_dim2_avx2(const int* data, const int* centroids, uint8_t* indices, int n, int k);

    void svt_ext_sad_calculation_8x8_16x16_avx2_intrin(uint8_t *src, uint32_t src_stride, uint8_t *ref,
        uint32_t ref_stride, uint32_t *p_best_sad_8x8,
        uint32_t *p_best_sad_16x16, uint32_t *p_best_mv8x8,
        uint32_t *p_best_mv16x16, uint32_t mv,
        uint32_t *p_sad16x16, uint32_t *p_sad8x8,
        Bool sub_sad);

    void svt_ext_sad_calculation_8x8_16x16_sse4_1_intrin(uint8_t *src, uint32_t src_stride, uint8_t *ref,
        uint32_t ref_stride, uint32_t *p_best_sad_8x8,
        uint32_t *p_best_sad_16x16, uint32_t *p_best_mv8x8,
        uint32_t *p_best_mv16x16, uint32_t mv,
        uint32_t *p_sad16x16, uint32_t *p_sad8x8,
        Bool sub_sad);

    void svt_ext_sad_calculation_32x32_64x64_sse4_intrin(uint32_t *p_sad16x16, uint32_t *p_best_sad_32x32,
        uint32_t *p_best_sad_64x64,
        uint32_t *p_best_mv32x32, uint32_t *p_best_mv64x64,
        uint32_t mv, uint32_t *p_sad32x32);
    void svt_ext_all_sad_calculation_8x8_16x16_avx2(uint8_t *src, uint32_t src_stride, uint8_t *ref,
        uint32_t ref_stride, uint32_t mv,
        uint8_t out_8x8,
        uint32_t *p_best_sad_8x8, uint32_t *p_best_sad_16x16,
        uint32_t *p_best_mv8x8, uint32_t *p_best_mv16x16,
        uint32_t p_eight_sad16x16[16][8],
        uint32_t p_eight_sad8x8[64][8], Bool sub_sad);
    void svt_ext_all_sad_calculation_8x8_16x16_sse4_1(uint8_t *src, uint32_t src_stride, uint8_t *ref,
        uint32_t ref_stride, uint32_t mv,
        uint8_t out_8x8,
        uint32_t *p_best_sad_8x8, uint32_t *p_best_sad_16x16,
        uint32_t *p_best_mv8x8, uint32_t *p_best_mv16x16,
        uint32_t p_eight_sad16x16[16][8],
        uint32_t p_eight_sad8x8[64][8], Bool sub_sad);
    void svt_ext_eight_sad_calculation_32x32_64x64_sse4_1(uint32_t  p_sad16x16[16][8],
        uint32_t *p_best_sad_32x32,
        uint32_t *p_best_sad_64x64,
        uint32_t *p_best_mv32x32, uint32_t *p_best_mv64x64,
        uint32_t mv, uint32_t p_sad32x32[4][8]);
    void svt_ext_eight_sad_calculation_32x32_64x64_avx2(uint32_t  p_sad16x16[16][8],
        uint32_t *p_best_sad_32x32,
        uint32_t *p_best_sad_64x64,
        uint32_t *p_best_mv32x32, uint32_t *p_best_mv64x64,
        uint32_t mv, uint32_t p_sad32x32[4][8]);
    uint32_t svt_compute4x_m_sad_avx2_intrin(
        const uint8_t *src, // input parameter, source samples Ptr
        uint32_t       src_stride, // input parameter, source stride
        const uint8_t *ref, // input parameter, reference samples Ptr
        uint32_t       ref_stride, // input parameter, reference stride
        uint32_t       height, // input parameter, block height (M)
        uint32_t       width); // input parameter, block width (N)
    void svt_initialize_buffer_32bits_sse2_intrin(uint32_t *pointer, uint32_t count128,
        uint32_t count32, uint32_t value);
    uint32_t svt_nxm_sad_kernel_sub_sampled_helper_avx2(const uint8_t *src, uint32_t src_stride,
        const uint8_t *ref, uint32_t ref_stride,
        uint32_t height, uint32_t width);
    uint32_t svt_nxm_sad_kernel_helper_avx2(const uint8_t *src, uint32_t src_stride, const uint8_t *ref,
        uint32_t ref_stride, uint32_t height, uint32_t width);
    uint32_t svt_nxm_sad_kernel_sub_sampled_helper_sse4_1(const uint8_t *src, uint32_t src_stride,
        const uint8_t *ref, uint32_t ref_stride,
        uint32_t height, uint32_t width);
    uint32_t svt_nxm_sad_kernel_helper_sse4_1(const uint8_t *src, uint32_t src_stride, const uint8_t *ref,
        uint32_t ref_stride, uint32_t height, uint32_t width);
    uint32_t nxm_sad_avg_kernel_helper_avx2(uint8_t *src, uint32_t src_stride, uint8_t *ref1,
        uint32_t ref1_stride, uint8_t *ref2, uint32_t ref2_stride,
        uint32_t height, uint32_t width);
    uint64_t svt_compute_mean_of_squared_values8x8_sse2_intrin(
        uint8_t* input_samples, // input parameter, input samples Ptr
        uint32_t input_stride, // input parameter, input stride
        uint32_t input_area_width, // input parameter, input area width
        uint32_t input_area_height); // input parameter, input area height
    uint64_t compute_subd_mean_of_squared_values8x8_sse2_intrin(
        uint8_t* input_samples, // input parameter, input samples Ptr
        uint16_t input_stride);
    uint64_t svt_compute_mean8x8_sse2_intrin(
        uint8_t* input_samples, // input parameter, input samples Ptr
        uint32_t input_stride, // input parameter, input stride
        uint32_t input_area_width, // input parameter, input area width
        uint32_t input_area_height); // input parameter, input area height
    uint64_t svt_compute_mean8x8_avx2_intrin(
        uint8_t *input_samples, // input parameter, input samples Ptr
        uint32_t input_stride, // input parameter, input stride
        uint32_t input_area_width, // input parameter, input area width
        uint32_t input_area_height);

    uint64_t svt_compute_sub_mean8x8_sse2_intrin(uint8_t* input_samples, uint16_t input_stride);

    void svt_compute_interm_var_four8x8_helper_sse2(uint8_t* input_samples, uint16_t input_stride,
        uint64_t* mean_of8x8_blocks, // mean of four  8x8
        uint64_t* mean_of_squared8x8_blocks); // meanSquared
    void svt_compute_interm_var_four8x8_avx2_intrin(uint8_t *input_samples, uint16_t input_stride,
        uint64_t *mean_of8x8_blocks, // mean of four  8x8
        uint64_t *mean_of_squared8x8_blocks);
    uint32_t sad_16bit_kernel_avx2(uint16_t *src, uint32_t src_stride, uint16_t *ref,
        uint32_t ref_stride, uint32_t height, uint32_t width);

    void svt_av1_apply_temporal_filter_planewise_fast_sse4_1(
            struct MeContext *context_ptr, const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre,
            int y_pre_stride, unsigned int block_width,
            unsigned int block_height, uint32_t *y_accum, uint16_t *y_count);
    void svt_av1_apply_temporal_filter_planewise_fast_avx2(
            struct MeContext *context_ptr, const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre,
            int y_pre_stride, unsigned int block_width,
            unsigned int block_height, uint32_t *y_accum, uint16_t *y_count);

    void svt_av1_apply_temporal_filter_planewise_fast_hbd_sse4_1(
        struct MeContext *context_ptr, const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre,
        int y_pre_stride, unsigned int block_width,
        unsigned int block_height, uint32_t *y_accum, uint16_t *y_count, uint32_t encoder_bit_depth);
    void svt_av1_apply_temporal_filter_planewise_fast_hbd_avx2(
        struct MeContext *context_ptr, const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre,
        int y_pre_stride, unsigned int block_width,
        unsigned int block_height, uint32_t *y_accum, uint16_t *y_count, uint32_t encoder_bit_depth);

    void svt_av1_apply_temporal_filter_planewise_medium_sse4_1(
        struct MeContext *context_ptr, const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre,
        int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src, int uv_src_stride,
        const uint8_t *u_pre, const uint8_t *v_pre, int uv_pre_stride, unsigned int block_width,
        unsigned int block_height, int ss_x, int ss_y, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
        uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count);
    void svt_av1_apply_temporal_filter_planewise_medium_avx2(
        struct MeContext *context_ptr, const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre,
        int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src, int uv_src_stride,
        const uint8_t *u_pre, const uint8_t *v_pre, int uv_pre_stride, unsigned int block_width,
        unsigned int block_height, int ss_x, int ss_y, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
        uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count);

    void svt_av1_apply_temporal_filter_planewise_medium_hbd_sse4_1(
        struct MeContext *context_ptr, const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre,
        int y_pre_stride, const uint16_t *u_src, const uint16_t *v_src, int uv_src_stride,
        const uint16_t *u_pre, const uint16_t *v_pre, int uv_pre_stride, unsigned int block_width,
        unsigned int block_height, int ss_x, int ss_y, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
        uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count, uint32_t encoder_bit_depth);
    void svt_av1_apply_temporal_filter_planewise_medium_hbd_avx2(
        struct MeContext *context_ptr, const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre,
        int y_pre_stride, const uint16_t *u_src, const uint16_t *v_src, int uv_src_stride,
        const uint16_t *u_pre, const uint16_t *v_pre, int uv_pre_stride, unsigned int block_width,
        unsigned int block_height, int ss_x, int ss_y, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
        uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count, uint32_t encoder_bit_depth);

    void svt_av1_apply_temporal_filter_planewise_sse4_1(
        struct MeContext *context_ptr, const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre,
        int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src, int uv_src_stride,
        const uint8_t *u_pre, const uint8_t *v_pre, int uv_pre_stride, unsigned int block_width,
        unsigned int block_height, int ss_x, int ss_y,
        uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
        uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count);
    void svt_av1_apply_temporal_filter_planewise_avx2(
        struct MeContext *context_ptr, const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre,
        int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src, int uv_src_stride,
        const uint8_t *u_pre, const uint8_t *v_pre, int uv_pre_stride, unsigned int block_width,
        unsigned int block_height, int ss_x, int ss_y,
        uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
        uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count);
    void svt_av1_apply_temporal_filter_planewise_hbd_sse4_1(
        struct MeContext *context_ptr, const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre,
        int y_pre_stride, const uint16_t *u_src, const uint16_t *v_src, int uv_src_stride,
        const uint16_t *u_pre, const uint16_t *v_pre, int uv_pre_stride, unsigned int block_width,
        unsigned int block_height, int ss_x, int ss_y,
        uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
        uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count, uint32_t encoder_bit_depth);
    void svt_av1_apply_temporal_filter_planewise_hbd_avx2(
        struct MeContext *context_ptr, const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre,
        int y_pre_stride, const uint16_t *u_src, const uint16_t *v_src, int uv_src_stride,
        const uint16_t *u_pre, const uint16_t *v_pre, int uv_pre_stride, unsigned int block_width,
        unsigned int block_height, int ss_x, int ss_y,
        uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
        uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count, uint32_t encoder_bit_depth);
    uint32_t variance_highbd_sse4_1(const uint16_t *a, int a_stride, const uint16_t *b, int b_stride,
                              int w, int h, uint32_t *sse);
    uint32_t variance_highbd_avx2(const uint16_t *a, int a_stride, const uint16_t *b, int b_stride,
                              int w, int h, uint32_t *sse);
    int svt_av1_haar_ac_sad_8x8_uint8_input_avx2(uint8_t *input, int stride, int hbd);

#endif

    /* Moved to aom_dsp_rtcd.c file:
    static void setup_rtcd_internal(EbAsm asm_type)
    */


#ifdef __cplusplus
}  // extern "C"
#endif

#endif
// clang-format on
