/*
 * Copyright(c) 2019 Netflix, Inc.
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#include "EbResize.h"
#include "EbUtility.h"
#include "EbSuperRes.h"
#include "EbIntraPrediction.h"

#define FILTER_BITS 7

// Calculates the scaled dimension given the original dimension and the scale
// denominator.
void calculate_scaled_size_helper(uint16_t *dim, uint8_t denom) {
    if (denom != SCALE_NUMERATOR) {
        // We need to ensure the constraint in "Appendix A" of the spec:
        // * FrameWidth is greater than or equal to 16
        // * FrameHeight is greater than or equal to 16
        // For this, we clamp the downscaled dimension to at least 16. One
        // exception: if original dimension itself was < 16, then we keep the
        // downscaled dimension to be same as the original, to ensure that resizing
        // is valid.
        const int min_dim = AOMMIN(16, *dim);
        // Use this version if we need *dim to be even
        // *width = (*width * SCALE_NUMERATOR + denom) / (2 * denom);
        // *width <<= 1;
        *dim = (uint16_t)((*dim * SCALE_NUMERATOR + denom / 2) / (denom));
        *dim = (uint16_t)AOMMAX(*dim, min_dim);
    }
}

static int32_t av1_get_upscale_convolve_step(int in_length, int out_length) {
    return ((in_length << RS_SCALE_SUBPEL_BITS) + out_length / 2) / out_length;
}

int32_t get_upscale_convolve_x0(int in_length, int out_length, int32_t x_step_qn) {
    const int     err = out_length * x_step_qn - (in_length << RS_SCALE_SUBPEL_BITS);
    const int32_t x0  = (-((out_length - in_length) << (RS_SCALE_SUBPEL_BITS - 1)) +
                        out_length / 2) /
            out_length +
        RS_SCALE_EXTRA_OFF - err / 2;
    return (int32_t)((uint32_t)x0 & RS_SCALE_SUBPEL_MASK);
}

static void av1_convolve_horiz_rs_c(const uint8_t *src, int src_stride, uint8_t *dst,
                                    int dst_stride, int w, int h, const int16_t *x_filters,
                                    int x0_qn, int x_step_qn) {
    src -= UPSCALE_NORMATIVE_TAPS / 2 - 1;
    for (int y = 0; y < h; ++y) {
        int x_qn = x0_qn;
        for (int x = 0; x < w; ++x) {
            const uint8_t *const src_x = &src[x_qn >> RS_SCALE_SUBPEL_BITS];
            const int x_filter_idx     = (x_qn & RS_SCALE_SUBPEL_MASK) >> RS_SCALE_EXTRA_BITS;
            assert(x_filter_idx <= RS_SUBPEL_MASK);
            const int16_t *const x_filter = &x_filters[x_filter_idx * UPSCALE_NORMATIVE_TAPS];
            int                  sum      = 0;
            for (int k = 0; k < UPSCALE_NORMATIVE_TAPS; ++k) sum += src_x[k] * x_filter[k];
            dst[x] = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
            x_qn += x_step_qn;
        }
        src += src_stride;
        dst += dst_stride;
    }
}

static void av1_highbd_convolve_horiz_rs_c(const uint16_t *src, int src_stride, uint16_t *dst,
                                           int dst_stride, int w, int h, const int16_t *x_filters,
                                           int x0_qn, int x_step_qn, int bd) {
    src -= UPSCALE_NORMATIVE_TAPS / 2 - 1;
    for (int y = 0; y < h; ++y) {
        int x_qn = x0_qn;
        for (int x = 0; x < w; ++x) {
            const uint16_t *const src_x = &src[x_qn >> RS_SCALE_SUBPEL_BITS];
            const int x_filter_idx      = (x_qn & RS_SCALE_SUBPEL_MASK) >> RS_SCALE_EXTRA_BITS;
            assert(x_filter_idx <= RS_SUBPEL_MASK);
            const int16_t *const x_filter = &x_filters[x_filter_idx * UPSCALE_NORMATIVE_TAPS];
            int                  sum      = 0;
            for (int k = 0; k < UPSCALE_NORMATIVE_TAPS; ++k) sum += src_x[k] * x_filter[k];
            dst[x] = clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
            x_qn += x_step_qn;
        }
        src += src_stride;
        dst += dst_stride;
    }
}

void upscale_normative_rect(const uint8_t *const input, int height, int width, int in_stride,
                            uint8_t *output, int height2, int width2, int out_stride, int x_step_qn,
                            int x0_qn, int pad_left, int pad_right) {
    assert(width > 0);
    assert(height > 0);
    assert(width2 > 0);
    assert(height2 > 0);
    assert(height2 == height);

    /* Extend the left/right pixels of the tile column if needed
    (either because we can't sample from other tiles, or because we're at
    a frame edge).
    Save the overwritten pixels into tmp_left and tmp_right.
    Note: Because we pass input-1 to av1_convolve_horiz_rs, we need one extra
    column of border pixels compared to what we'd naively think.*/
    const int      border_cols = UPSCALE_NORMATIVE_TAPS / 2 + 1;
    uint8_t       *tmp_left    = NULL;
    uint8_t       *tmp_right   = NULL;
    uint8_t *const in_tl       = (uint8_t *)(input - border_cols);
    uint8_t *const in_tr       = (uint8_t *)(input + width);

    if (pad_left) {
        tmp_left = (uint8_t *)svt_aom_malloc(sizeof(*tmp_left) * border_cols * height);
        for (int i = 0; i < height; i++) {
            svt_memcpy(tmp_left + i * border_cols, in_tl + i * in_stride, border_cols);
            memset(in_tl + i * in_stride, input[i * in_stride], border_cols);
        }
    }
    if (pad_right) {
        tmp_right = (uint8_t *)svt_aom_malloc(sizeof(*tmp_right) * border_cols * height);
        for (int i = 0; i < height; i++) {
            svt_memcpy(tmp_right + i * border_cols, in_tr + i * in_stride, border_cols);
            memset(in_tr + i * in_stride, input[i * in_stride + width - 1], border_cols);
        }
    }

    av1_convolve_horiz_rs_c(input - 1,
                            in_stride,
                            output,
                            out_stride,
                            width2,
                            height2,
                            &av1_resize_filter_normative[0][0],
                            x0_qn,
                            x_step_qn);

    /* Restore the left/right border pixels */
    if (pad_left) {
        for (int i = 0; i < height; i++) {
            svt_memcpy(in_tl + i * in_stride, tmp_left + i * border_cols, border_cols);
        }
        svt_aom_free(tmp_left);
    }
    if (pad_right) {
        for (int i = 0; i < height; i++) {
            svt_memcpy(in_tr + i * in_stride, tmp_right + i * border_cols, border_cols);
        }
        svt_aom_free(tmp_right);
    }
}

void highbd_upscale_normative_rect(const uint8_t *const input, int height, int width, int in_stride,
                                   uint8_t *output, int height2, int width2, int out_stride,
                                   int x_step_qn, int x0_qn, int pad_left, int pad_right, int bd) {
    assert(width > 0);
    assert(height > 0);
    assert(width2 > 0);
    assert(height2 > 0);
    assert(height2 == height);

    /* Extend the left/right pixels of the tile column if needed
    (either because we can't sample from other tiles, or because we're at
    a frame edge).
    Save the overwritten pixels into tmp_left and tmp_right.
    Note: Because we pass input-1 to av1_convolve_horiz_rs, we need one extra
    column of border pixels compared to what we'd naively think.*/
    const int       border_cols = UPSCALE_NORMATIVE_TAPS / 2 + 1;
    const int       border_size = border_cols * sizeof(uint16_t);
    uint16_t       *tmp_left    = NULL;
    uint16_t       *tmp_right   = NULL;
    uint16_t *const input16     = (uint16_t *)input; //CONVERT_TO_SHORTPTR(input);
    uint16_t *const in_tl       = input16 - border_cols;
    uint16_t *const in_tr       = input16 + width;
    if (pad_left) {
        tmp_left = (uint16_t *)svt_aom_malloc(sizeof(*tmp_left) * border_cols * height);
        for (int i = 0; i < height; i++) {
            svt_memcpy(tmp_left + i * border_cols, in_tl + i * in_stride, border_size);
            svt_aom_memset16(in_tl + i * in_stride, input16[i * in_stride], border_cols);
        }
    }
    if (pad_right) {
        tmp_right = (uint16_t *)svt_aom_malloc(sizeof(*tmp_right) * border_cols * height);
        for (int i = 0; i < height; i++) {
            svt_memcpy(tmp_right + i * border_cols, in_tr + i * in_stride, border_size);
            svt_aom_memset16(
                in_tr + i * in_stride, input16[i * in_stride + width - 1], border_cols);
        }
    }

    av1_highbd_convolve_horiz_rs_c(((uint16_t *)(input)-1),
                                   in_stride,
                                   (uint16_t *)(output),
                                   out_stride,
                                   width2,
                                   height2,
                                   &av1_resize_filter_normative[0][0],
                                   x0_qn,
                                   x_step_qn,
                                   bd);

    /*Restore the left/right border pixels*/
    if (pad_left) {
        for (int i = 0; i < height; i++) {
            svt_memcpy(in_tl + i * in_stride, tmp_left + i * border_cols, border_size);
        }
        svt_aom_free(tmp_left);
    }
    if (pad_right) {
        for (int i = 0; i < height; i++) {
            svt_memcpy(in_tr + i * in_stride, tmp_right + i * border_cols, border_size);
        }
        svt_aom_free(tmp_right);
    }
}

void svt_av1_upscale_normative_rows(const Av1Common *cm, const uint8_t *src, int src_stride,
                                    uint8_t *dst, int dst_stride, int rows, int sub_x, int bd,
                                    Bool is_16bit_pipeline) {
    int       high_bd                = bd > EB_8BIT || is_16bit_pipeline;
    const int downscaled_plane_width = ROUND_POWER_OF_TWO(cm->frm_size.frame_width, sub_x);
    const int upscaled_plane_width   = ROUND_POWER_OF_TWO(cm->frm_size.superres_upscaled_width,
                                                        sub_x);
    const int superres_denom         = cm->frm_size.superres_denominator;

    TileInfo      tile_col;
    const int32_t x_step_qn = av1_get_upscale_convolve_step(downscaled_plane_width,
                                                            upscaled_plane_width);
    int32_t       x0_qn     = get_upscale_convolve_x0(
        downscaled_plane_width, upscaled_plane_width, x_step_qn);
    for (int j = 0; j < cm->tiles_info.tile_cols; j++) {
        svt_av1_tile_set_col(&tile_col, &cm->tiles_info, cm->mi_cols, j);

        /*Determine the limits of this tile column in both the source
        and destination images.
        Note: The actual location which we start sampling from is
        (downscaled_x0 - 1 + (x0_qn/2^14)), and this quantity increases
        by exactly dst_width * (x_step_qn/2^14) pixels each iteration.*/
        const int downscaled_x0 = tile_col.mi_col_start << (MI_SIZE_LOG2 - sub_x);
        const int downscaled_x1 = tile_col.mi_col_end << (MI_SIZE_LOG2 - sub_x);
        const int src_width     = downscaled_x1 - downscaled_x0;

        const int upscaled_x0 = (downscaled_x0 * superres_denom) / SCALE_NUMERATOR;
        int       upscaled_x1;
        if (j == cm->tiles_info.tile_cols - 1) {
            /*Note that we can't just use AOMMIN here - due to rounding,
            (downscaled_x1 * superres_denom) / SCALE_NUMERATOR may be less than
            upscaled_plane_width.*/
            upscaled_x1 = upscaled_plane_width;
        } else
            upscaled_x1 = (downscaled_x1 * superres_denom) / SCALE_NUMERATOR;

        const uint8_t *const src_ptr   = src + (downscaled_x0 << high_bd);
        uint8_t *const       dst_ptr   = dst + (upscaled_x0 << high_bd);
        const int            dst_width = upscaled_x1 - upscaled_x0;

        const int pad_left  = (j == 0);
        const int pad_right = (j == cm->tiles_info.tile_cols - 1);

        if (high_bd)
            highbd_upscale_normative_rect(src_ptr,
                                          rows,
                                          src_width,
                                          src_stride,
                                          dst_ptr,
                                          rows,
                                          dst_width,
                                          dst_stride,
                                          x_step_qn,
                                          x0_qn,
                                          pad_left,
                                          pad_right,
                                          bd);
        else
            upscale_normative_rect(src_ptr,
                                   rows,
                                   src_width,
                                   src_stride,
                                   dst_ptr,
                                   rows,
                                   dst_width,
                                   dst_stride,
                                   x_step_qn,
                                   x0_qn,
                                   pad_left,
                                   pad_right);

        /*Update the fractional pixel offset to prepare for the next tile col*/
        x0_qn += (dst_width * x_step_qn) - (src_width << RS_SCALE_SUBPEL_BITS);
    }
}
