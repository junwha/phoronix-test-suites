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

#ifndef EbPredictionUnit_h
#define EbPredictionUnit_h

#include "EbDefinitions.h"
#include "EbSyntaxElements.h"
#include "EbDefinitions.h"
#include "EbMotionVectorUnit.h"
#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(push, 1)
typedef struct PredictionUnit {
    Mv      mv[MAX_NUM_OF_REF_PIC_LIST]; // 16-bytes
    uint8_t inter_pred_direction_index;

    // Intra Mode
#if CLN_CAND_TYPES
    int8_t angle_delta[PLANE_TYPES];
    UvPredictionMode intra_chroma_mode;
#else
    int32_t  angle_delta[PLANE_TYPES];
#if !CLN_REMOVE_REDUND_6
    Bool   is_directional_mode_flag;
    Bool   is_directional_chroma_mode_flag;
#endif
    uint32_t intra_chroma_mode;
#endif
    // Inter Mode
#if !CLN_REMOVE_REDUND
    Bool     is_compound;
#endif
    uint8_t    ref_frame_type;
    MotionMode motion_mode;
    uint16_t   num_proj_ref;
    uint32_t   overlappable_neighbors[2];
#if CLN_CAND_TYPES
    uint8_t cfl_alpha_idx; // Index of the alpha Cb and alpha Cr combination
    uint8_t cfl_alpha_signs; // Joint sign of alpha Cb and alpha Cr
#else
    // Index of the alpha Cb and alpha Cr combination
    int32_t cfl_alpha_idx;
    // Joint sign of alpha Cb and alpha Cr
    int32_t cfl_alpha_signs;
#endif
} PredictionUnit;
#pragma pack(pop)

#ifdef __cplusplus
}
#endif
#endif //EbPredictionUnit_h
