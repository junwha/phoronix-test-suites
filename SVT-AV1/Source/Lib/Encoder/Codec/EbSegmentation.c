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

#include "EbSegmentation.h"
#include "EbSegmentationParams.h"
#include "EbMotionEstimationContext.h"
#include "common_dsp_rtcd.h"

uint16_t get_variance_for_cu(const BlockGeom *blk_geom, uint16_t *variance_ptr) {
    int index0, index1;
    //Assumes max CU size is 64
    switch (blk_geom->bsize) {
    case BLOCK_4X4:
    case BLOCK_4X8:
    case BLOCK_8X4:
    case BLOCK_8X8:
        index0 = index1 = ME_TIER_ZERO_PU_8x8_0 + ((blk_geom->origin_x >> 3) + blk_geom->origin_y);
        break;

    case BLOCK_8X16:
        index0 = ME_TIER_ZERO_PU_8x8_0 + ((blk_geom->origin_x >> 3) + blk_geom->origin_y);
        index1 = index0 + 1;
        break;

    case BLOCK_16X8:
        index0 = ME_TIER_ZERO_PU_8x8_0 + ((blk_geom->origin_x >> 3) + blk_geom->origin_y);
        index1 = index0 + blk_geom->origin_y;
        break;

    case BLOCK_4X16:
    case BLOCK_16X4:
    case BLOCK_16X16:
        index0 = index1 = ME_TIER_ZERO_PU_16x16_0 +
            ((blk_geom->origin_x >> 4) + (blk_geom->origin_y >> 2));
        break;

    case BLOCK_16X32:
        index0 = ME_TIER_ZERO_PU_16x16_0 + ((blk_geom->origin_x >> 4) + (blk_geom->origin_y >> 2));
        index1 = index0 + 1;
        break;

    case BLOCK_32X16:
        index0 = ME_TIER_ZERO_PU_16x16_0 + ((blk_geom->origin_x >> 4) + (blk_geom->origin_y >> 2));
        index1 = index0 + (blk_geom->origin_y >> 2);
        break;

    case BLOCK_8X32:
    case BLOCK_32X8:
    case BLOCK_32X32:
        index0 = index1 = ME_TIER_ZERO_PU_32x32_0 +
            ((blk_geom->origin_x >> 5) + (blk_geom->origin_y >> 4));
        break;

    case BLOCK_32X64:
        index0 = ME_TIER_ZERO_PU_32x32_0 + ((blk_geom->origin_x >> 5) + (blk_geom->origin_y >> 4));
        index1 = index0 + 1;
        break;

    case BLOCK_64X32:
        index0 = ME_TIER_ZERO_PU_32x32_0 + ((blk_geom->origin_x >> 5) + (blk_geom->origin_y >> 4));
        index1 = index0 + (blk_geom->origin_y >> 4);
        break;

    case BLOCK_64X64:
    case BLOCK_16X64:
    case BLOCK_64X16:
    default: index0 = index1 = 0; break;
    }
    return (variance_ptr[index0] + variance_ptr[index1]) >> 1;
}

void apply_segmentation_based_quantization(const BlockGeom *blk_geom, PictureControlSet *pcs_ptr,
                                           SuperBlock *sb_ptr, BlkStruct *blk_ptr) {
    uint16_t           *variance_ptr        = pcs_ptr->parent_pcs_ptr->variance[sb_ptr->index];
    SegmentationParams *segmentation_params = &pcs_ptr->parent_pcs_ptr->frm_hdr.segmentation_params;
    uint16_t            variance            = get_variance_for_cu(blk_geom, variance_ptr);
    for (int i = 0; i < MAX_SEGMENTS; i++) {
        if (variance <= segmentation_params->variance_bin_edge[i]) {
            blk_ptr->segment_id = i;
            break;
        }
    }
    int32_t q_index = pcs_ptr->parent_pcs_ptr->frm_hdr.quantization_params.base_q_idx +
        pcs_ptr->parent_pcs_ptr->frm_hdr.segmentation_params
            .feature_data[blk_ptr->segment_id][SEG_LVL_ALT_Q];
    blk_ptr->qindex = q_index;
}

void setup_segmentation(PictureControlSet *pcs_ptr, SequenceControlSet *scs_ptr) {
    SegmentationParams *segmentation_params = &pcs_ptr->parent_pcs_ptr->frm_hdr.segmentation_params;
    segmentation_params->segmentation_enabled =
        (Bool)(scs_ptr->static_config.enable_adaptive_quantization == 1);
    if (segmentation_params->segmentation_enabled) {
        segmentation_params->segmentation_update_data =
            1; //always updating for now. Need to set this based on actual deltas
        segmentation_params->segmentation_update_map = 1;
        segmentation_params->segmentation_temporal_update =
            FALSE; //!(pcs_ptr->parent_pcs_ptr->av1FrameType == KEY_FRAME || pcs_ptr->parent_pcs_ptr->av1FrameType == INTRA_ONLY_FRAME);
        find_segment_qps(segmentation_params, pcs_ptr);
        for (int i = 0; i < MAX_SEGMENTS; i++)
            segmentation_params->feature_enabled[i][SEG_LVL_ALT_Q] = 1;

        calculate_segmentation_data(segmentation_params);
    }
}

void calculate_segmentation_data(SegmentationParams *segmentation_params) {
    for (int i = 0; i < MAX_SEGMENTS; i++) {
        for (int j = 0; j < SEG_LVL_MAX; j++) {
            if (segmentation_params->feature_enabled[i][j]) {
                segmentation_params->last_active_seg_id = i;
                if (j >= SEG_LVL_REF_FRAME) {
                    segmentation_params->seg_id_pre_skip = 1;
                }
            }
        }
    }
}

void find_segment_qps(SegmentationParams *segmentation_params,
                      PictureControlSet  *pcs_ptr) { //QP needs to be specified as qpindex, not qp.

    uint16_t    min_var = UINT16_MAX, max_var = MIN_UNSIGNED_VALUE, avg_var = 0;
    const float strength = 2; //to tune

    // get range of variance
    for (uint32_t sb_idx = 0; sb_idx < pcs_ptr->sb_total_count; ++sb_idx) {
        uint16_t *variance_ptr = pcs_ptr->parent_pcs_ptr->variance[sb_idx];
        uint32_t  var_index, local_avg = 0;
        // Loop over all 8x8s in a 64x64
        for (var_index = ME_TIER_ZERO_PU_8x8_0; var_index <= ME_TIER_ZERO_PU_8x8_63; var_index++) {
            max_var = MAX(max_var, variance_ptr[var_index]);
            min_var = MIN(min_var, variance_ptr[var_index]);
            local_avg += variance_ptr[var_index];
        }
        avg_var += (local_avg >> 6);
    }
    avg_var /= pcs_ptr->sb_total_count;
    avg_var = svt_log2f(avg_var);

    //get variance bin edges & QPs
    uint16_t min_var_log = svt_log2f(MAX(1, min_var));
    uint16_t max_var_log = svt_log2f(MAX(1, max_var));
    uint16_t step_size   = (uint16_t)(max_var_log - min_var_log) <= MAX_SEGMENTS
          ? 1
          : ROUND(((max_var_log - min_var_log)) / MAX_SEGMENTS);
    uint16_t bin_edge    = min_var_log + step_size;
    uint16_t bin_center  = bin_edge >> 1;

    for (int i = 0; i < MAX_SEGMENTS; i++) {
        segmentation_params->variance_bin_edge[i]           = POW2(bin_edge);
        segmentation_params->feature_data[i][SEG_LVL_ALT_Q] = ROUND((uint16_t)strength *
                                                                    (MAX(1, bin_center) - avg_var));
        bin_edge += step_size;
        bin_center += step_size;
    }
}
