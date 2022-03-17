/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_ENCODER_PASS2_STRATEGY_H_
#define AOM_AV1_ENCODER_PASS2_STRATEGY_H_

#include "level.h"
#include "encoder.h"
#include "Av1Common.h"

#ifdef __cplusplus
extern "C" {
#endif
#define FRAMES_TO_CHECK_DECAY 8
#define KF_MIN_FRAME_BOOST 80.0
#define KF_MAX_FRAME_BOOST 128.0
#define MIN_KF_BOOST 600 // Minimum boost for non-static KF interval
#define MAX_KF_BOOST 3200
#define MIN_STATIC_KF_BOOST 5400 // Minimum boost for static KF interval
#define MAX_KF_BOOST_LOW_KI 3000 // Maximum boost for KF with low interval
#define MAX_KF_BOOST_HIGHT_KI 5000 // Maximum boost for KF with hight interval
#define KF_INTERVAL_TH 64 // Low/high KF interval threshold
// structure of accumulated stats and features in a gf group
typedef struct {
    double     gf_group_err;
    StatStruct gf_stat_struct;
    double     gf_group_raw_error;
    double     gf_group_skip_pct;
    double     gf_group_inactive_zone_rows;

    double decay_accumulator;
    double zero_motion_accumulator;
    double this_frame_mv_in_out;
} GF_GROUP_STATS;

void svt_av1_init_second_pass(struct SequenceControlSet *scs_ptr);
void svt_av1_init_single_pass_lap(struct SequenceControlSet *scs_ptr);
void svt_av1_new_framerate(struct SequenceControlSet *scs_ptr, double framerate);
void find_init_qp_middle_pass(struct SequenceControlSet      *scs_ptr,
                              struct PictureParentControlSet *pcs_ptr);
void one_pass_rt_rate_alloc(struct PictureParentControlSet *pcs_ptr);
void process_rc_stat(struct PictureParentControlSet *pcs_ptr);

void        svt_av1_twopass_postencode_update(struct PictureParentControlSet *ppcs_ptr);
extern void crf_assign_max_rate(PictureParentControlSet *ppcs_ptr);
extern void set_rc_param(struct SequenceControlSet *scs_ptr);
int         frame_is_kf_gf_arf(PictureParentControlSet *ppcs_ptr);
#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOM_AV1_ENCODER_PASS2_STRATEGY_H_
