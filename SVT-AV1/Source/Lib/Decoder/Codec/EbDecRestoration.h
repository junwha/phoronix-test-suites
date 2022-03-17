/*
* Copyright(c) 2019 Netflix, Inc.
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef EbDecRestoration_h
#define EbDecRestoration_h

#ifdef __cplusplus
extern "C" {
#endif

#include "EbDecHandle.h"

/* The LR_PAD_SIZE must be 3 for both luma and chroma
   The value is set to 6 to keep the functionality for
   all planes when subsampling is applied in the pad functions */
#define LR_PAD_SIDE 6
#define LR_PAD_MAX (LR_PAD_SIDE << 1)

void dec_av1_loop_restoration_save_boundary_lines(EbDecHandle *dec_handle, int after_cdef);
void lr_pad_pic(EbPictureBufferDesc *recon_picture_buf, FrameHeader *frame_hdr,
                EbColorConfig *color_cfg);
void dec_av1_loop_restoration_filter_frame(EbDecHandle *dec_handle, int optimized_lr,
                                           int enable_flag);
void dec_av1_loop_restoration_filter_row(EbDecHandle *dec_handle, int32_t sb_row,
                                         uint8_t **rec_buff, int *rec_stride,
                                         Av1PixelRect *tile_rect, int optimized_lr, uint8_t *dst,
                                         int thread_cnt);
#ifdef __cplusplus
}
#endif

#endif
