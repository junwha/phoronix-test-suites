/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_ENCODER_FIRSTPASS_H_
#define AOM_AV1_ENCODER_FIRSTPASS_H_

#include "EbDefinitions.h"
#include "EbRateControlProcess.h"
#include "EbPictureControlSet.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TURN_OFF_EC_FIRST_PASS 1

#define FORCED_BLK_SIZE 16
#define FIRST_PASS_Q 10.0

#define DOUBLE_DIVIDE_CHECK(x) ((x) < 0 ? (x)-0.000001 : (x) + 0.000001)

#define MAX_LAG_BUFFERS 35
#define MIN_ZERO_MOTION 0.95
#define MAX_SR_CODED_ERROR 40
#define MAX_RAW_ERR_VAR 2000
#define MIN_MV_IN_OUT 0.4

#define VLOW_MOTION_THRESHOLD 950

// size of firstpass macroblocks in terms of MIs.
#define FP_MIB_SIZE 4
#define FP_MIB_SIZE_LOG2 2

// The maximum duration of a GF group that is static (e.g. a slide show).
#define MAX_STATIC_GF_GROUP_LENGTH 250
#define MAX_LAP_BUFFERS 35

/*!
 * \brief The stucture of acummulated frame stats in the first pass.
 */
typedef struct {
    /*!
   * Frame number in display order, if stats are for a single frame.
   * No real meaning for a collection of frames.
   */
    double frame;
    /*!
   * Weight assigned to this frame (or total weight for the collection of
   * frames) currently based on intra factor and brightness factor. This is used
   * to distribute bits betweeen easier and harder frames.
   */
    double weight;
    /*!
   * Intra prediction error.
   */
    double intra_error;
    /*!
   * Best of intra pred error and inter pred error using last frame as ref.
   */
    double coded_error;
    /*!
   * Best of intra pred error and inter pred error using golden frame as ref.
   */
    double sr_coded_error;
    /*!
   * Percentage of blocks with inter pred error < intra pred error.
   */
    double pcnt_inter;
    /*!
   * Percentage of blocks using (inter prediction and) non-zero motion vectors.
   */
    double pcnt_motion;
    /*!
   * Percentage of blocks where golden frame was better than last or intra:
   * inter pred error using golden frame < inter pred error using last frame and
   * inter pred error using golden frame < intra pred error
   */
    double pcnt_second_ref;
    /*!
   * Percentage of blocks where intra and inter prediction errors were very
   * close. Note that this is a 'weighted count', that is, the so blocks may be
   * weighted by how close the two errors were.
   */
    double pcnt_neutral;
    /*!
   * Percentage of blocks that have almost no intra error residual
   * (i.e. are in effect completely flat and untextured in the intra
   * domain). In natural videos this is uncommon, but it is much more
   * common in animations, graphics and screen content, so may be used
   * as a signal to detect these types of content.
   */
    double intra_skip_pct;
    /*!
   * Image mask rows top and bottom.
   */
    double inactive_zone_rows;
    /*!
   * Image mask columns at left and right edges.
   */
    double inactive_zone_cols;
    /*!
   * Mean of absolute value of row motion vectors.
   */
    double mvr_abs;
    /*!
   * Mean of absolute value of column motion vectors.
   */
    double mvc_abs;
    /*!
   * Value in range [-1,1] indicating fraction of row and column motion vectors
   * that point inwards (negative MV value) or outwards (positive MV value).
   * For example, value of 1 indicates, all row/column MVs are inwards.
   */
    double mv_in_out_count;
    /*!
   * Duration of the frame / collection of frames.
   */
    double duration;
    /*!
   * 1.0 if stats are for a single frame, OR
   * Number of frames in this collection for which the stats are accumulated.
   */
    double     count;
    StatStruct stat_struct;
} FIRSTPASS_STATS;

/*!\cond */

#define FC_ANIMATION_THRESH 0.15
enum {
    FC_NORMAL             = 0,
    FC_GRAPHICS_ANIMATION = 1,
    FRAME_CONTENT_TYPES   = 2
} UENUM1BYTE(FRAME_CONTENT_TYPE);

typedef struct {
    unsigned char             index;
    /*frame_update_type*/ int update_type[MAX_STATIC_GF_GROUP_LENGTH];
#if !FRFCTR_RC_P1
    unsigned char             frame_disp_idx[MAX_STATIC_GF_GROUP_LENGTH];
#endif

    // TODO(jingning): Unify the data structure used here after the new control
    // mechanism is in place.
    int layer_depth[MAX_STATIC_GF_GROUP_LENGTH];
#if !FRFCTR_RC_P1
    int arf_boost[MAX_STATIC_GF_GROUP_LENGTH];
#endif
    int max_layer_depth;
    int max_layer_depth_allowed;
    int bit_allocation[MAX_STATIC_GF_GROUP_LENGTH];
    int size;
} GF_GROUP;

typedef struct {
    FIRSTPASS_STATS *stats_in_start;
    // used when writing the stat.i.e in the first pass
    FIRSTPASS_STATS *stats_in_end_write;
    FIRSTPASS_STATS *stats_in_end;
    FIRSTPASS_STATS *stats_in_buf_end;
    FIRSTPASS_STATS *total_stats;
    FIRSTPASS_STATS *total_left_stats;
    int64_t          last_frame_accumulated;
} STATS_BUFFER_CTX;

/*!\endcond */

/*!
 * \brief Two pass status and control data.
 */
typedef struct {
#if !FRFCTR_RC_P1
    /*!\cond */
    unsigned int section_intra_rating;
#endif
    // Circular queue of first pass stats stored for most recent frames.
    // cpi->output_pkt_list[i].data.twopass_stats.buf points to actual data stored
    // here.
    const FIRSTPASS_STATS *stats_in;
    STATS_BUFFER_CTX      *stats_buf_ctx;
    int                    first_pass_done;
    int64_t                bits_left;
    double                 modified_error_min;
    double                 modified_error_max;
    double                 modified_error_left;
#if !FRFCTR_RC_P1
    // An indication of the content type of the current frame
    FRAME_CONTENT_TYPE fr_content_type;
#endif

    // Projected total bits available for a key frame group of frames
    int64_t kf_group_bits;

    // Error score of frames still to be coded in kf group
    int64_t kf_group_error_left;

    int     kf_zeromotion_pct;
    int     extend_minq;
    int     extend_maxq;
    int     extend_minq_fast;
    uint8_t passes;
    /*!\endcond */
} TWO_PASS;

/*!\cond */

// This structure contains several key parameters to be accumulated for this
// frame.
typedef struct {
    // Intra prediction error.
    int64_t intra_error;
    // Best of intra pred error and inter pred error using last frame as ref.
    int64_t coded_error;
    // Best of intra pred error and inter pred error using golden frame as ref.
    int64_t sr_coded_error;
    // Count of motion vector.
    int mv_count;
    // Count of blocks that pick inter prediction (inter pred error is smaller
    // than intra pred error).
    int inter_count;
    // Count of blocks that pick second ref (golden frame).
    int second_ref_count;
    // Count of blocks where the inter and intra are very close and very low.
    double neutral_count;
    // Count of blocks where intra error is very small.
    int intra_skip_count;
    // Start row.
    int image_data_start_row;
    // Count of unique non-zero motion vectors.
    // Sum of inward motion vectors.
    int sum_in_vectors;
    // Sum of motion vector row.
    // Sum of motion vector column.
    int sum_mvc;
    // Sum of absolute value of motion vector row.
    int sum_mvr_abs;
    // Sum of absolute value of motion vector column.
    int sum_mvc_abs;
    // A factor calculated using intra pred error.
    double intra_factor;
    // A factor that measures brightness.
    double brightness_factor;
} FRAME_STATS;

// This structure contains first pass data.
typedef struct {
    // Buffer holding frame stats for all MACROBLOCKs.
    // mb_stats[i] stores the FRAME_STATS of the ith
    // MB in raster scan order.
    FRAME_STATS *mb_stats;
    // Buffer to store the prediction error of the (0,0) motion
    // vector using the last source frame as the reference.
    // raw_motion_err_list[i] stores the raw_motion_err of
    // the ith MB in raster scan order.
    int *raw_motion_err_list;
} FirstPassData;
#if !FRFCTR_RC_P8
struct EncodeFrameParams;
#endif
struct AV1EncoderConfig;
struct TileDataEnc;

void svt_av1_twopass_zero_stats(FIRSTPASS_STATS *section);
void svt_av1_accumulate_stats(FIRSTPASS_STATS *section, const FIRSTPASS_STATS *frame);
/*!\endcond */

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOM_AV1_ENCODER_FIRSTPASS_H_
