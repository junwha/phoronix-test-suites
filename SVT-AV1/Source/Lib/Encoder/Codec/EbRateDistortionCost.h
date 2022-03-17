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

#ifndef EbRateDistortionCost_h
#define EbRateDistortionCost_h

/***************************************
 * Includes
 ***************************************/
#include "EbModeDecisionProcess.h"
#include "EbEncIntraPrediction.h"
#include "EbEncInterPrediction.h"
#include "EbLambdaRateTables.h"
#include "EbTransforms.h"
#include "EbEncDecProcess.h"
#include "EbEntropyCoding.h"

#ifdef __cplusplus
extern "C" {
#endif
extern uint64_t svt_av1_cost_coeffs_txb(struct ModeDecisionContext *ctx, uint8_t allow_update_cdf,
                                        FRAME_CONTEXT                      *ec_ctx,
                                        struct ModeDecisionCandidateBuffer *candidate_buffer_ptr,
                                        const TranLow *const qcoeff, uint16_t eob,
                                        PlaneType plane_type, TxSize transform_size,
                                        TxType transform_type, int16_t txb_skip_ctx,
                                        int16_t dc_sign_ctx, Bool reduced_transform_set_flag);

extern void coding_loop_context_generation(PictureControlSet   *pcs_ptr,
                                           ModeDecisionContext *context_ptr, BlkStruct *blk_ptr,
                                           uint32_t blk_origin_x, uint32_t blk_origin_y,
                                           NeighborArrayUnit *skip_coeff_neighbor_array,
                                           NeighborArrayUnit *leaf_partition_neighbor_array);
#if !CLN_MOVE_COSTS_2
extern EbErrorType av1_txb_calc_cost(
    ModeDecisionCandidate *candidate_ptr, // input parameter, prediction result Ptr
    int16_t                txb_skip_ctx,
    uint32_t               txb_index, // input parameter, TU index inside the CU
    uint32_t
        y_count_non_zero_coeffs, // input parameter, number of non zero Y quantized coefficients
    uint32_t
        cb_count_non_zero_coeffs, // input parameter, number of non zero cb quantized coefficients
    uint32_t
        cr_count_non_zero_coeffs, // input parameter, number of non zero cr quantized coefficients
    uint64_t y_txb_distortion
        [DIST_CALC_TOTAL], // input parameter, Y distortion for both Normal and Cbf zero modes
    uint64_t cb_txb_distortion
        [DIST_CALC_TOTAL], // input parameter, Cb distortion for both Normal and Cbf zero modes
    uint64_t cr_txb_distortion
        [DIST_CALC_TOTAL], // input parameter, Cr distortion for both Normal and Cbf zero modes
    COMPONENT_TYPE component_type,
    uint64_t      *y_txb_coeff_bits, // input parameter, Y quantized coefficients rate
    uint64_t      *cb_txb_coeff_bits, // input parameter, Cb quantized coefficients rate
    uint64_t      *cr_txb_coeff_bits, // input parameter, Cr quantized coefficients rate
    TxSize         txsize,
    uint64_t       lambda); // input parameter, lambda for Luma

extern EbErrorType txb_calc_cost(
    uint32_t cu_size, ModeDecisionCandidate *candidate_ptr, uint32_t txb_index,
    uint32_t transform_size, uint32_t transform_chroma_size, uint32_t y_count_non_zero_coeffs,
    uint32_t cb_count_non_zero_coeffs, uint32_t cr_count_non_zero_coeffs,
    uint64_t y_txb_distortion[DIST_CALC_TOTAL], uint64_t cb_txb_distortion[DIST_CALC_TOTAL],
    uint64_t cr_txb_distortion[DIST_CALC_TOTAL], uint32_t component_mask,
    uint64_t *y_txb_coeff_bits, uint64_t *cb_txb_coeff_bits, uint64_t *cr_txb_coeff_bits,
    uint32_t qp, uint64_t lambda, uint64_t lambda_chroma);
#endif
extern EbErrorType intra_luma_mode_context(BlkStruct *blk_ptr, uint32_t luma_mode,
                                           int32_t *prediction_index);
extern EbErrorType intra2_nx2_n_fast_cost_islice(
    BlkStruct *blk_ptr, struct ModeDecisionCandidateBuffer *candidate_buffer_ptr, uint32_t qp,
    uint64_t luma_distortion, uint64_t chroma_distortion, uint64_t lambda,
    PictureControlSet *pcs_ptr);
extern EbErrorType merge_skip_full_cost(
    SuperBlock *sb_ptr, BlkStruct *blk_ptr, uint32_t cu_size, uint32_t cu_size_log2,
    ModeDecisionCandidateBuffer *candidate_buffer_ptr, uint32_t qp, uint64_t *y_distortion,
    uint64_t *cb_distortion, uint64_t *cr_distortion, uint64_t lambda, uint64_t lambda_chroma,
    uint64_t *y_coeff_bits, uint64_t *cb_coeff_bits, uint64_t *cr_coeff_bits,
    uint32_t transform_size, uint32_t transform_chroma_size, PictureControlSet *pcs_ptr);
extern EbErrorType split_flag_rate(ModeDecisionContext *context_ptr, BlkStruct *blk_ptr,
                                   uint32_t split_flag, uint64_t *split_rate, uint64_t lambda,
                                   MdRateEstimationContext *md_rate_estimation_ptr,
                                   uint32_t                 tb_max_depth);

#define RDDIV_BITS 7

#define RDCOST(RM, R, D)                                                         \
    (ROUND_POWER_OF_TWO(((int64_t)(R)) * ((int64_t)(RM)), AV1_PROB_COST_SHIFT) + \
     ((int64_t)(D) * ((int64_t)1 << RDDIV_BITS)))

extern EbErrorType av1_split_flag_rate(PictureParentControlSet *pcs_ptr,
                                       ModeDecisionContext *context_ptr, BlkStruct *blk_ptr,
                                       uint32_t leaf_index, PartitionType partitionType,
                                       uint64_t *split_rate, uint64_t lambda,
                                       MdRateEstimationContext *md_rate_estimation_ptr,
                                       uint32_t                 tb_max_depth);
extern uint64_t    av1_intra_fast_cost(struct ModeDecisionContext *context_ptr, BlkStruct *blk_ptr,
#if CLN_MOVE_COSTS
                                       ModeDecisionCandidateBuffer *candidate_buffer, uint32_t qp,
#else
                                       ModeDecisionCandidate *candidate_ptr, uint32_t qp,
#endif
                                       uint64_t luma_distortion, uint64_t chroma_distortion,
                                       uint64_t lambda, PictureControlSet *pcs_ptr,
                                       CandidateMv *ref_mv_stack, const BlockGeom *blk_geom,
                                       uint32_t miRow, uint32_t miCol, uint8_t enable_inter_intra,
                                       uint32_t left_neighbor_mode, uint32_t top_neighbor_mode);

extern uint64_t av1_inter_fast_cost(struct ModeDecisionContext *context_ptr, BlkStruct *blk_ptr,
#if CLN_MOVE_COSTS
                                    ModeDecisionCandidateBuffer *candidate_buffer, uint32_t qp,
#else
                                    ModeDecisionCandidate *candidate_ptr, uint32_t qp,
#endif
                                    uint64_t luma_distortion, uint64_t chroma_distortion,
                                    uint64_t lambda, PictureControlSet *pcs_ptr,
                                    CandidateMv *ref_mv_stack, const BlockGeom *blk_geom,
                                    uint32_t miRow, uint32_t miCol, uint8_t enable_inter_intra,
                                    uint32_t left_neighbor_mode, uint32_t top_neighbor_mode);

EbErrorType        av1_full_cost_light_pd0(ModeDecisionContext                *context_ptr,
                                           struct ModeDecisionCandidateBuffer *candidate_buffer_ptr,
                                           uint64_t *y_distortion, uint64_t lambda,
                                           uint64_t *y_coeff_bits);
extern EbErrorType av1_intra_full_cost(PictureControlSet *pcs_ptr, ModeDecisionContext *context_ptr,
                                       struct ModeDecisionCandidateBuffer *candidate_buffer_ptr,
                                       BlkStruct *blk_ptr, uint64_t *y_distortion,
                                       uint64_t *cb_distortion, uint64_t *cr_distortion,
                                       uint64_t lambda, uint64_t *y_coeff_bits,
                                       uint64_t *cb_coeff_bits, uint64_t *cr_coeff_bits,
                                       BlockSize bsize);

extern EbErrorType av1_inter_full_cost(PictureControlSet *pcs_ptr, ModeDecisionContext *context_ptr,
                                       struct ModeDecisionCandidateBuffer *candidate_buffer_ptr,
                                       BlkStruct *blk_ptr, uint64_t *y_distortion,
                                       uint64_t *cb_distortion, uint64_t *cr_distortion,
                                       uint64_t lambda, uint64_t *y_coeff_bits,
                                       uint64_t *cb_coeff_bits, uint64_t *cr_coeff_bits,
                                       BlockSize bsize);
extern uint64_t    get_tx_size_bits(ModeDecisionCandidateBuffer *candidateBuffer,
                                    ModeDecisionContext *context_ptr, PictureControlSet *pcs_ptr,
                                    uint8_t tx_depth, Bool block_has_coeff);

MvJointType svt_av1_get_mv_joint(const MV *mv);

static INLINE uint8_t av1_drl_ctx(const CandidateMv *ref_mv_stack, int32_t ref_idx) {
    return ref_mv_stack[ref_idx].weight >= REF_CAT_LEVEL
        ? ref_mv_stack[ref_idx + 1].weight >= REF_CAT_LEVEL ? 0 : 1
        : ref_mv_stack[ref_idx + 1].weight < REF_CAT_LEVEL ? 2
                                                           : 0;
}
#ifdef __cplusplus
}
#endif
#endif //EbRateDistortionCost_h
