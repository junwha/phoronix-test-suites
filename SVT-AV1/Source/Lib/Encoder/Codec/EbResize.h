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

#ifndef EbResize_h
#define EbResize_h

#include "EbDefinitions.h"
#include "EbPictureBufferDesc.h"
#include "EbInterPrediction.h"
#include "EbSequenceControlSet.h"
#include "EbSuperRes.h"
#include "EbReferenceObject.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t encoding_width;
    uint16_t encoding_height;
    uint8_t  superres_denom;
} superres_params_type;

void scale_source_references(SequenceControlSet *scs_ptr, PictureParentControlSet *pcs_ptr,
                             EbPictureBufferDesc *input_picture_ptr);

void scale_rec_references(PictureControlSet *pcs_ptr, EbPictureBufferDesc *input_picture_ptr,
                          uint8_t hbd_mode_decision);

void use_scaled_rec_refs_if_needed(PictureControlSet   *pcs_ptr,
                                   EbPictureBufferDesc *input_picture_ptr,
                                   EbReferenceObject *ref_obj, EbPictureBufferDesc **ref_pic,
                                   uint8_t hbd_mode_decision);

void use_scaled_source_refs_if_needed(PictureParentControlSet *pcs_ptr,
                                      EbPictureBufferDesc     *input_picture_ptr,
                                      EbPaReferenceObject     *ref_obj,
                                      EbPictureBufferDesc    **ref_pic_ptr,
                                      EbPictureBufferDesc    **quarter_ref_pic_ptr,
                                      EbPictureBufferDesc    **sixteenth_ref_pic_ptr);

void scale_pcs_params(SequenceControlSet *scs_ptr, PictureParentControlSet *pcs_ptr,
                      superres_params_type spr_params, uint16_t source_width,
                      uint16_t source_height);

void init_resize_picture(SequenceControlSet *scs_ptr, PictureParentControlSet *pcs_ptr);

void reset_resized_picture(SequenceControlSet *scs_ptr, PictureParentControlSet *pcs_ptr,
                           EbPictureBufferDesc *input_picture_ptr);

static INLINE int coded_to_superres_mi(int mi_col, int denom) {
    return (mi_col * denom + SCALE_NUMERATOR / 2) / SCALE_NUMERATOR;
}

#define filteredinterp_filters1000 av1_resize_filter_normative

#ifdef __cplusplus
}
#endif
#endif // EbResize_h
