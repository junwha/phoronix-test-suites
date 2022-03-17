/*
* Copyright(c) 2019 Intel Corporation
* Copyright (c) 2019, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef EbIntraPrediction_h
#define EbIntraPrediction_h

#include "EbSvtAv1.h"
#include "EbObject.h"
#include "EbBlockStructures.h"

#ifdef __cplusplus
extern "C" {
#endif
#define MAX_PU_SIZE 64

struct ModeDecisionContext;

typedef void (*IntraPredFnC)(uint8_t *dst, ptrdiff_t stride, int32_t w, int32_t h,
                             const uint8_t *above, const uint8_t *left);
typedef void (*IntraHighBdPredFnC)(uint16_t *dst, ptrdiff_t stride, int32_t w, int32_t h,
                                   const uint16_t *above, const uint16_t *left, int32_t bd);

typedef void (*IntraPredFn)(uint8_t *dst, ptrdiff_t stride, const uint8_t *above,
                            const uint8_t *left);

typedef void (*IntraHighPredFn)(uint16_t *dst, ptrdiff_t stride, const uint16_t *above,
                                const uint16_t *left, int32_t bd);

typedef struct IntraReferenceSamples {
    EbDctor  dctor;
    uint8_t *cb_intra_reference_array;
    uint8_t *cr_intra_reference_array;
    uint8_t *y_intra_filtered_reference_array;

    uint8_t *y_intra_reference_array_reverse;
    uint8_t *y_intra_filtered_reference_array_reverse;
    uint8_t *cb_intra_reference_array_reverse;
    uint8_t *cr_intra_reference_array_reverse;

    // Scratch buffers used in the interpolaiton process
    uint8_t reference_above_line_y[(MAX_PU_SIZE << 2) + 1];
    uint8_t reference_left_line_y[(MAX_PU_SIZE << 2) + 1];
    Bool  above_ready_flag_y;
    Bool  left_ready_flag_y;

    uint8_t reference_above_line_cb[(MAX_PU_SIZE << 2) + 2];
    uint8_t reference_left_line_cb[(MAX_PU_SIZE << 2) + 2];
    Bool  above_ready_flag_cb;
    Bool  left_ready_flag_cb;

    uint8_t reference_above_line_cr[(MAX_PU_SIZE << 2) + 2];
    uint8_t reference_left_line_cr[(MAX_PU_SIZE << 2) + 2];
    Bool  above_ready_flag_cr;
    Bool  left_ready_flag_cr;
} IntraReferenceSamples;

typedef struct IntraReference16bitSamples {
    EbDctor   dctor;
    uint16_t *cb_intra_reference_array;
    uint16_t *cr_intra_reference_array;
    uint16_t *y_intra_filtered_reference_array;

    uint16_t *y_intra_reference_array_reverse;
    uint16_t *y_intra_filtered_reference_array_reverse;
    uint16_t *cb_intra_reference_array_reverse;
    uint16_t *cr_intra_reference_array_reverse;

    // Scratch buffers used in the interpolaiton process
    uint16_t reference_above_line_y[(MAX_PU_SIZE << 2) + 1];
    uint16_t reference_left_line_y[(MAX_PU_SIZE << 2) + 1];
    Bool   above_ready_flag_y;
    Bool   left_ready_flag_y;

    uint16_t reference_above_line_cb[(MAX_PU_SIZE << 2) + 2];
    uint16_t reference_left_line_cb[(MAX_PU_SIZE << 2) + 2];
    Bool   above_ready_flag_cb;
    Bool   left_ready_flag_cb;

    uint16_t reference_above_line_cr[(MAX_PU_SIZE << 2) + 2];
    uint16_t reference_left_line_cr[(MAX_PU_SIZE << 2) + 2];
    Bool   above_ready_flag_cr;
    Bool   left_ready_flag_cr;
} IntraReference16bitSamples;

#define TOTAL_LUMA_MODES 35
#define TOTAL_CHROMA_MODES 5
#define TOTAL_INTRA_GROUPS 5
#define INTRA_PLANAR_MODE 0
#define INTRA_DC_MODE 1
#define INTRA_HORIZONTAL_MODE 10
#define INTRA_VERTICAL_MODE 26
#define STRONG_INTRA_SMOOTHING_BLOCKSIZE 32
#define SMOOTHING_THRESHOLD 8
#define SMOOTHING_THRESHOLD_10BIT 32

/////####.... For recursive intra prediction.....#####///

#define FILTER_INTRA_SCALE_BITS 4
extern const int8_t eb_av1_filter_intra_taps[FILTER_INTRA_MODES][8][8];

/////####.... To make functions common between EbIntraPrediction.c &
void *svt_aom_memset16(void *dest, int32_t val, size_t length);

int32_t use_intra_edge_upsample(int32_t bs0, int32_t bs1, int32_t delta, int32_t type);

BlockSize scale_chroma_bsize(BlockSize bsize, int32_t subsampling_x, int32_t subsampling_y);

int32_t intra_edge_filter_strength(int32_t bs0, int32_t bs1, int32_t delta, int32_t type);

enum {
    NEED_LEFT       = 1 << 1,
    NEED_ABOVE      = 1 << 2,
    NEED_ABOVERIGHT = 1 << 3,
    NEED_ABOVELEFT  = 1 << 4,
    NEED_BOTTOMLEFT = 1 << 5,
};

static const int32_t mode_to_angle_map[] = {
    0,
    90,
    180,
    45,
    135,
    113,
    157,
    203,
    67,
    0,
    0,
    0,
    0,
};

int                  is_smooth(const BlockModeInfoEnc *mbmi, int plane);
int                  is_smooth_dec(const BlockModeInfo *mbmi, int plane);
extern const uint8_t extend_modes[INTRA_MODES];

/* TODO: Need to harmonize with fun from EbAdaptiveMotionVectorPrediction.c */
int32_t intra_has_top_right(BlockSize sb_size, BlockSize bsize, int32_t mi_row, int32_t mi_col,
                            int32_t top_available, int32_t right_available, PartitionType partition,
                            TxSize txsz, int32_t row_off, int32_t col_off, int32_t ss_x,
                            int32_t ss_y);

extern int32_t intra_has_bottom_left(BlockSize sb_size, BlockSize bsize, int32_t mi_row,
                                     int32_t mi_col, int32_t bottom_available,
                                     int32_t left_available, PartitionType partition, TxSize txsz,
                                     int32_t row_off, int32_t col_off, int32_t ss_x, int32_t ss_y);

extern IntraPredFn eb_pred[INTRA_MODES][TX_SIZES_ALL];
extern IntraPredFn dc_pred[2][2][TX_SIZES_ALL];

extern IntraHighPredFn pred_high[INTRA_MODES][TX_SIZES_ALL];
extern IntraHighPredFn dc_pred_high[2][2][TX_SIZES_ALL];

void dr_predictor(uint8_t *dst, ptrdiff_t stride, TxSize tx_size, const uint8_t *above,
                  const uint8_t *left, int32_t upsample_above, int32_t upsample_left,
                  int32_t angle);

void filter_intra_edge_corner(uint8_t *p_above, uint8_t *p_left);

void highbd_dr_predictor(uint16_t *dst, ptrdiff_t stride, TxSize tx_size, const uint16_t *above,
                         const uint16_t *left, int32_t upsample_above, int32_t upsample_left,
                         int32_t angle, int32_t bd);

void filter_intra_edge_corner_high(uint16_t *p_above, uint16_t *p_left);

void highbd_filter_intra_predictor(uint16_t *dst, ptrdiff_t stride, TxSize tx_size,
                                   const uint16_t *above, const uint16_t *left, int mode, int bd);

typedef void (*EB_INTRA_NOANG_TYPE)(const uint32_t size, uint8_t *ref_samples,
                                    uint8_t       *prediction_ptr,
                                    const uint32_t prediction_buffer_stride, const Bool skip);
typedef void (*EB_INTRA_DC_AV1_TYPE)(
    const uint32_t size, //input parameter, denotes the size of the current PU
    uint8_t       *ref_samples, //input parameter, pointer to the reference samples
    uint8_t       *dst, //output parameter, pointer to the prediction
    const uint32_t
        prediction_buffer_stride, //input parameter, denotes the stride for the prediction ptr
    const Bool skip); //skip half rows
typedef void (*EB_INTRA_NOANG_16bit_TYPE)(const uint32_t size, uint16_t *ref_samples,
                                          uint16_t      *prediction_ptr,
                                          const uint32_t prediction_buffer_stride,
                                          const Bool   skip);
typedef void (*EB_INTRA_ANG_Z1_Z2_Z3_16bit_TYPE)(const uint32_t size, uint16_t *ref_samples,
                                                 uint16_t      *dst,
                                                 const uint32_t prediction_buffer_stride,
                                                 const Bool skip, uint16_t dx, uint16_t dy,
                                                 uint16_t bd);
typedef void (*EB_INTRA_ANG_TYPE)(uint32_t size, uint8_t *ref_samp_main, uint8_t *prediction_ptr,
                                  uint32_t prediction_buffer_stride, const Bool skip,
                                  int32_t intra_pred_angle);
typedef void (*EB_INTRA_ANG_16BIT_TYPE)(
    uint32_t  size, //input parameter, denotes the size of the current PU
    uint16_t *ref_samp_main, //input parameter, pointer to the reference samples
    uint16_t *prediction_ptr, //output parameter, pointer to the prediction
    uint32_t  prediction_buffer_stride, //input parameter, denotes the stride for the prediction ptr
    const Bool skip, int32_t intra_pred_angle);

extern void svt_cfl_luma_subsampling_420_lbd_c(const uint8_t *input, // AMIR-> Changed to 8 bit
                                               int32_t input_stride, int16_t *output_q3,
                                               int32_t width, int32_t height);
extern void svt_cfl_luma_subsampling_420_hbd_c(const uint16_t *input, int32_t input_stride,
                                               int16_t *output_q3, int32_t width, int32_t height);
extern void svt_subtract_average_c(int16_t *pred_buf_q3, int32_t width, int32_t height,
                                   int32_t round_offset, int32_t num_pel_log2);

//CFL_PREDICT_FN(c, lbd)

void svt_cfl_predict_lbd_c(const int16_t *pred_buf_q3,
                           uint8_t       *pred, // AMIR ADDED
                           int32_t        pred_stride,
                           uint8_t       *dst, // AMIR changed to 8 bit
                           int32_t dst_stride, int32_t alpha_q3, int32_t bit_depth, int32_t width,
                           int32_t height);

void svt_cfl_predict_hbd_c(const int16_t *pred_buf_q3,
                           uint16_t      *pred, // AMIR ADDED
                           int32_t        pred_stride,
                           uint16_t      *dst, // AMIR changed to 8 bit
                           int32_t dst_stride, int32_t alpha_q3, int32_t bit_depth, int32_t width,
                           int32_t height);

static INLINE int32_t cfl_idx_to_alpha(int32_t alpha_idx, int32_t joint_sign,
                                       CflPredType pred_type) {
    const int32_t alpha_sign = (pred_type == CFL_PRED_U) ? CFL_SIGN_U(joint_sign)
                                                         : CFL_SIGN_V(joint_sign);
    if (alpha_sign == CFL_SIGN_ZERO)
        return 0;
    const int32_t abs_alpha_q3 = (pred_type == CFL_PRED_U) ? CFL_IDX_U(alpha_idx)
                                                           : CFL_IDX_V(alpha_idx);
    return (alpha_sign == CFL_SIGN_POS) ? abs_alpha_q3 + 1 : -abs_alpha_q3 - 1;
}

extern void        filter_intra_edge(OisMbResults *ois_mb_results_ptr, uint8_t mode,
                                     uint16_t max_frame_width, uint16_t max_frame_height, int32_t p_angle,
                                     int32_t cu_origin_x, int32_t cu_origin_y, uint8_t *above_row,
                                     uint8_t *left_col);
extern EbErrorType intra_prediction_open_loop_mb(int32_t p_angle, uint8_t ois_intra_mode,
                                                 uint32_t srcOriginX, uint32_t srcOriginY,
                                                 TxSize tx_size, uint8_t *above_row,
                                                 uint8_t *left_col, uint8_t *dst,
                                                 uint32_t dst_stride);
/* Function pointers return by CfL functions */
typedef void (*CflSubtractAverageFn)(int16_t *dst);

CflSubtractAverageFn svt_get_subtract_average_fn_c(TxSize tx_size);
#define get_subtract_average_fn svt_get_subtract_average_fn_c

// Declare a size-specific wrapper for the size-generic function. The compiler
// will inline the size generic function in here, the advantage is that the size
// will be constant allowing for loop unrolling and other constant propagated
// goodness.
#define CFL_SUB_AVG_X(arch, width, height, round_offset, num_pel_log2)               \
    void svt_subtract_average_##width##x##height##_##arch(int16_t *buf) {            \
        svt_subtract_average_##arch(buf, width, height, round_offset, num_pel_log2); \
    }

// Declare size-specific wrappers for all valid CfL sizes.
#define CFL_SUB_AVG_FN(arch)                                                  \
    CFL_SUB_AVG_X(arch, 4, 4, 8, 4)                                           \
    CFL_SUB_AVG_X(arch, 4, 8, 16, 5)                                          \
    CFL_SUB_AVG_X(arch, 4, 16, 32, 6)                                         \
    CFL_SUB_AVG_X(arch, 8, 4, 16, 5)                                          \
    CFL_SUB_AVG_X(arch, 8, 8, 32, 6)                                          \
    CFL_SUB_AVG_X(arch, 8, 16, 64, 7)                                         \
    CFL_SUB_AVG_X(arch, 8, 32, 128, 8)                                        \
    CFL_SUB_AVG_X(arch, 16, 4, 32, 6)                                         \
    CFL_SUB_AVG_X(arch, 16, 8, 64, 7)                                         \
    CFL_SUB_AVG_X(arch, 16, 16, 128, 8)                                       \
    CFL_SUB_AVG_X(arch, 16, 32, 256, 9)                                       \
    CFL_SUB_AVG_X(arch, 32, 8, 128, 8)                                        \
    CFL_SUB_AVG_X(arch, 32, 16, 256, 9)                                       \
    CFL_SUB_AVG_X(arch, 32, 32, 512, 10)                                      \
    CflSubtractAverageFn svt_get_subtract_average_fn_##arch(TxSize tx_size) { \
        const CflSubtractAverageFn sub_avg[TX_SIZES_ALL] = {                  \
            svt_subtract_average_4x4_##arch, /* 4x4 */                        \
            svt_subtract_average_8x8_##arch, /* 8x8 */                        \
            svt_subtract_average_16x16_##arch, /* 16x16 */                    \
            svt_subtract_average_32x32_##arch, /* 32x32 */                    \
            NULL, /* 64x64 (invalid CFL size) */                              \
            svt_subtract_average_4x8_##arch, /* 4x8 */                        \
            svt_subtract_average_8x4_##arch, /* 8x4 */                        \
            svt_subtract_average_8x16_##arch, /* 8x16 */                      \
            svt_subtract_average_16x8_##arch, /* 16x8 */                      \
            svt_subtract_average_16x32_##arch, /* 16x32 */                    \
            svt_subtract_average_32x16_##arch, /* 32x16 */                    \
            NULL, /* 32x64 (invalid CFL size) */                              \
            NULL, /* 64x32 (invalid CFL size) */                              \
            svt_subtract_average_4x16_##arch, /* 4x16 (invalid CFL size) */   \
            svt_subtract_average_16x4_##arch, /* 16x4 (invalid CFL size) */   \
            svt_subtract_average_8x32_##arch, /* 8x32 (invalid CFL size) */   \
            svt_subtract_average_32x8_##arch, /* 32x8 (invalid CFL size) */   \
            NULL, /* 16x64 (invalid CFL size) */                              \
            NULL, /* 64x16 (invalid CFL size) */                              \
        };                                                                    \
        /* Modulo TX_SIZES_ALL to ensure that an attacker won't be able to */ \
        /* index the function pointer array out of bounds. */                 \
        return sub_avg[tx_size % TX_SIZES_ALL];                               \
    }

static INLINE int32_t av1_is_directional_mode(PredictionMode mode) {
    return mode >= V_PRED && mode <= D67_PRED;
}

static INLINE int get_palette_bsize_ctx(BlockSize bsize) {
    return num_pels_log2_lookup[bsize] - num_pels_log2_lookup[BLOCK_8X8];
}

static INLINE Bool av1_use_angle_delta(BlockSize bsize, uint8_t enable_angle_delta) {
    return (enable_angle_delta ? bsize >= BLOCK_8X8 : (Bool)enable_angle_delta);
}

#ifdef __cplusplus
}
#endif
#endif // EbIntraPrediction_h
