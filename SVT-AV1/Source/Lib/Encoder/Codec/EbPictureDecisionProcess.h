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

#ifndef EbPictureDecision_h
#define EbPictureDecision_h

#include "EbDefinitions.h"
#include "EbPictureControlSet.h"
#include "EbSequenceControlSet.h"
#include "EbUtility.h"

/***************************************
 * Extern Function Declaration
 ***************************************/
EbErrorType picture_decision_context_ctor(EbThreadContext   *thread_context_ptr,
                                          const EbEncHandle *enc_handle_ptr);

extern void *picture_decision_kernel(void *input_ptr);

void downsample_decimation_input_picture(PictureParentControlSet *pcs_ptr,
                                         EbPictureBufferDesc     *inputPaddedPicturePtr,
                                         EbPictureBufferDesc     *quarterDecimatedPicturePtr,
                                         EbPictureBufferDesc     *sixteenthDecimatedPicturePtr);

void pad_picture_to_multiple_of_min_blk_size_dimensions(SequenceControlSet  *scs_ptr,
                                                        EbPictureBufferDesc *input_picture_ptr);
void pad_picture_to_multiple_of_min_blk_size_dimensions_16bit(
    SequenceControlSet *scs_ptr, EbPictureBufferDesc *input_picture_ptr);
void picture_pre_processing_operations(PictureParentControlSet *pcs_ptr,
                                       SequenceControlSet      *scs_ptr);
void pad_picture_to_multiple_of_sb_dimensions(EbPictureBufferDesc *input_padded_picture_ptr);

void gathering_picture_statistics(SequenceControlSet *scs_ptr, PictureParentControlSet *pcs_ptr,
                                  EbPictureBufferDesc *input_picture_ptr,
                                  EbPictureBufferDesc *input_padded_picture_ptr,
                                  EbPictureBufferDesc *sixteenth_decimated_picture_ptr,
                                  uint32_t             sb_total_count);

void down_sample_chroma(EbPictureBufferDesc *input_picture_ptr,
                        EbPictureBufferDesc *outputPicturePtr);

Bool is_delayed_intra(PictureParentControlSet *pcs);

/**************************************
 * Context
 **************************************/
typedef struct PictureDecisionContext {
    EbDctor  dctor;
    EbFifo  *picture_analysis_results_input_fifo_ptr;
    EbFifo  *picture_decision_results_output_fifo_ptr;
    EbFifo  *me_fifo_ptr;
    uint64_t last_solid_color_frame_poc;

    Bool     reset_running_avg;
    int8_t     tf_motion_direction; // -1: invalid   0: horz  1: vert
    uint32_t **ahd_running_avg_cb;
    uint32_t **ahd_running_avg_cr;
    uint32_t **ahd_running_avg;
    Bool     is_scene_change_detected;
    uint8_t    transition_present;
    // Dynamic GOP
    uint32_t ttl_region_activity_cost[MAX_NUMBER_OF_REGIONS_IN_WIDTH]
                                     [MAX_NUMBER_OF_REGIONS_IN_HEIGHT];

    uint32_t total_number_of_mini_gops;

    uint32_t mini_gop_start_index[MINI_GOP_WINDOW_MAX_COUNT];
    uint32_t mini_gop_end_index[MINI_GOP_WINDOW_MAX_COUNT];
    uint32_t mini_gop_length[MINI_GOP_WINDOW_MAX_COUNT];
    uint32_t mini_gop_intra_count[MINI_GOP_WINDOW_MAX_COUNT];
    uint32_t mini_gop_idr_count[MINI_GOP_WINDOW_MAX_COUNT];
    uint32_t mini_gop_hierarchical_levels[MINI_GOP_WINDOW_MAX_COUNT];
    Bool   mini_gop_activity_array[MINI_GOP_MAX_COUNT];
    uint32_t mini_gop_region_activity_cost_array[MINI_GOP_MAX_COUNT][MAX_NUMBER_OF_REGIONS_IN_WIDTH]
                                                [MAX_NUMBER_OF_REGIONS_IN_HEIGHT];

    uint32_t mini_gop_group_faded_in_pictures_count[MINI_GOP_MAX_COUNT];
    uint32_t mini_gop_group_faded_out_pictures_count[MINI_GOP_MAX_COUNT];
    uint8_t  lay0_toggle; //3 way toggle 0->1->2
    uint8_t  lay1_toggle; //2 way toggle 0->1
    uint8_t  lay2_toggle; //2 way toggle 0->1
    Bool
        mini_gop_toggle; //mini GOP toggling since last Key Frame  K-0-1-0-1-0-K-0-1-0-1-K-0-1.....
    uint8_t                  last_i_picture_sc_class0;
    uint8_t                  last_i_picture_sc_class1;
    uint8_t                  last_i_picture_sc_class2;
    uint64_t                 key_poc;
    uint8_t                  tf_level;
    uint32_t                 tf_pic_arr_cnt;
    PictureParentControlSet *tf_pic_array[1 << MAX_TEMPORAL_LAYERS];
    PictureParentControlSet *mg_pictures_array[1 << MAX_TEMPORAL_LAYERS];
    DepCntPicInfo            updated_links_arr
        [UPDATED_LINKS]; //if not empty, this picture is a depn-cnt-cleanUp triggering picture (I frame; or MG size change )
    //this array will store all others pictures needing a dep-cnt clean up.
    uint32_t
        other_updated_links_cnt; //how many other pictures in the above array needing a dep-cnt clean-up
    PictureParentControlSet *prev_delayed_intra; //Key frame or I of LDP short MG
    uint32_t                 mg_size; //number of active pictures in above array
    PictureParentControlSet *mg_pictures_array_disp_order[1 << MAX_TEMPORAL_LAYERS];

    int64_t mg_progress_id;

    int32_t last_i_noise_levels_log1p_fp16[MAX_MB_PLANE];
    double  last_i_noise_levels[MAX_MB_PLANE];
} PictureDecisionContext;

#endif // EbPictureDecision_h
