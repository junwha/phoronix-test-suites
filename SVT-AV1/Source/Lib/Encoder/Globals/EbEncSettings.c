/*
* Copyright(c) 2022 Intel Corporation
*
* This source code is subject to the terms of the BSD 3-Clause Clear License and
* the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/
// SUMMARY
//   Contains the encoder settings API functions

/**************************************
 * Includes
 **************************************/
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "EbVersion.h"
#include "EbDefinitions.h"
#include "EbSvtAv1Enc.h"
#include "EbSvtAv1Metadata.h"
#include "EbEncSettings.h"

#include "EbLog.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#endif

/******************************************
* Verify Settings
******************************************/
EbErrorType svt_av1_verify_settings(SequenceControlSet *scs_ptr) {
    EbErrorType               return_error   = EB_ErrorNone;
    EbSvtAv1EncConfiguration *config         = &scs_ptr->static_config;
    unsigned int              channel_number = config->channel_id;
    if (config->enc_mode > MAX_ENC_PRESET) {
        SVT_ERROR("Instance %u: EncoderMode must be in the range of [0-%d]\n",
                  channel_number + 1,
                  MAX_ENC_PRESET);
        return_error = EB_ErrorBadParameter;
    }
    if (config->enc_mode == MAX_ENC_PRESET) {
        SVT_WARN(
            "EncoderMode (preset): %d was developed for the sole purpose of debugging and or "
            "running fast convex-hull encoding. This configuration should not be used for any "
            "benchmarking or quality analysis\n",
            MAX_ENC_PRESET);
    }
    if (scs_ptr->max_input_luma_width < 64) {
        SVT_ERROR("Instance %u: Source Width must be at least 64\n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    if (scs_ptr->max_input_luma_height < 64) {
        SVT_ERROR("Instance %u: Source Height must be at least 64\n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    if (config->pred_structure > 2) {
        SVT_ERROR("Instance %u: Pred Structure must be [0, 1 or 2]\n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    if (scs_ptr->max_input_luma_width % 8 &&
        scs_ptr->static_config.compressed_ten_bit_format == 1) {
        SVT_ERROR(
            "Instance %u: Only multiple of 8 width is supported for compressed 10-bit inputs \n",
            channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (scs_ptr->max_input_luma_width > 16384) {
        SVT_ERROR("Instance %u: Source Width must be less than or equal to 16384\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (scs_ptr->max_input_luma_height > 8704) {
        SVT_ERROR("Instance %u: Source Height must be less than or equal to 8704)\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->level != 0 && (config->level < 20 || config->level > 73)) {
        SVT_ERROR("Instance %u: Level must be in the range of [2.0-7.3]\n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->qp > MAX_QP_VALUE) {
        SVT_ERROR("Instance %u: %s must be [0 - %d]\n",
                  channel_number + 1,
                  config->enable_tpl_la ? "CRF" : "QP",
                  MAX_QP_VALUE);
        return_error = EB_ErrorBadParameter;
    }
    if (config->hierarchical_levels > 5) {
        SVT_ERROR("Instance %u: Hierarchical Levels supported [0-5]\n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    if ((config->intra_period_length < -2 || config->intra_period_length > 2 * ((1 << 30) - 1)) &&
        config->rate_control_mode == 0) {
        SVT_ERROR("Instance %u: The intra period must be [-2, 2^31-2]  \n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
#if FTR_CBR
    if ((config->intra_period_length < 0) && config->rate_control_mode == 1) {
#else
    if ((config->intra_period_length < 0) && config->rate_control_mode >= 1) {
#endif
        SVT_ERROR("Instance %u: The intra period must be > 0 for RateControlMode %d \n",
                  channel_number + 1,
                  config->rate_control_mode);
        return_error = EB_ErrorBadParameter;
    }

    if (config->intra_refresh_type > 2 || config->intra_refresh_type < 1) {
        SVT_ERROR("Instance %u: Invalid intra Refresh Type [1-2]\n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->enable_dlf_flag > 1) {
        SVT_ERROR("Instance %u: Invalid LoopFilterEnable. LoopFilterEnable must be [0 - 1]\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->rate_control_mode > 2 &&
        (config->pass == ENC_FIRST_PASS || config->rc_stats_buffer.buf)) {
        SVT_ERROR("Instance %u: Only rate control mode 0~2 are supported for 2-pass \n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    if (config->profile > 2) {
        SVT_ERROR("Instance %u: The maximum allowed profile value is 2 \n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    // Check if the current input video is conformant with the Level constraint
    if (config->frame_rate > (240 << 16)) {
        SVT_ERROR("Instance %u: The maximum allowed frame rate is 240 fps\n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    // Check that the frame_rate is non-zero
    if (!config->frame_rate) {
        SVT_ERROR("Instance %u: The frame rate should be greater than 0 fps \n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->recode_loop > 4) {
        SVT_ERROR("Instance %u: The recode_loop must be [0 - 4] \n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    if (config->rate_control_mode > 2) {
        SVT_ERROR("Instance %u: The rate control mode must be [0 - 2] \n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    if (config->look_ahead_distance > MAX_LAD && config->look_ahead_distance != (uint32_t)~0) {
        SVT_ERROR(
            "Instance %u: The lookahead distance must be [0 - %d] \n", channel_number + 1, MAX_LAD);

        return_error = EB_ErrorBadParameter;
    }
    if ((unsigned)config->tile_rows > 6 || (unsigned)config->tile_columns > 6) {
        SVT_ERROR("Instance %u: Log2Tile rows/cols must be [0 - 6] \n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    if ((1u << config->tile_rows) * (1u << config->tile_columns) > 128 ||
        config->tile_columns > 4) {
        SVT_ERROR("Instance %u: MaxTiles is 128 and MaxTileCols is 16 (Annex A.3) \n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    if (config->restricted_motion_vector > 1) {
        SVT_ERROR("Instance %u : Invalid Restricted Motion Vector flag [0 - 1]\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->max_qp_allowed > MAX_QP_VALUE) {
        SVT_ERROR("Instance %u: MaxQpAllowed must be [1 - %d]\n", channel_number + 1, MAX_QP_VALUE);
        return_error = EB_ErrorBadParameter;
    } else if (config->min_qp_allowed >= MAX_QP_VALUE) {
        SVT_ERROR(
            "Instance %u: MinQpAllowed must be [1 - %d]\n", channel_number + 1, MAX_QP_VALUE - 1);
        return_error = EB_ErrorBadParameter;
    } else if ((config->min_qp_allowed) > (config->max_qp_allowed)) {
        SVT_ERROR("Instance %u:  MinQpAllowed must be smaller than MaxQpAllowed\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    } else if ((config->min_qp_allowed) == 0) {
        SVT_ERROR("Instance %u: MinQpAllowed must be [1 - %d]. Lossless coding not supported\n",
                  channel_number + 1,
                  MAX_QP_VALUE - 1);
        return_error = EB_ErrorBadParameter;
    }
    if (config->use_qp_file > 1) {
        SVT_ERROR("Instance %u : Invalid use_qp_file. use_qp_file must be [0 - 1]\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->stat_report > 1) {
        SVT_ERROR("Instance %u : Invalid StatReport. StatReport must be [0 - 1]\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->high_dynamic_range_input > 1) {
        SVT_ERROR(
            "Instance %u : Invalid HighDynamicRangeInput. HighDynamicRangeInput must be [0 - 1]\n",
            channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->screen_content_mode > 2) {
        SVT_ERROR(
            "Instance %u : Invalid screen_content_mode. screen_content_mode must be [0 - 2]\n",
            channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    // IntraBC
    if (scs_ptr->intrabc_mode > 3 || scs_ptr->intrabc_mode < -1) {
        SVT_ERROR("Instance %u: Invalid intraBC mode [0-3, -1 for default], your input: %i\n",
                  channel_number + 1,
                  scs_ptr->intrabc_mode);
        return_error = EB_ErrorBadParameter;
    }

    if (scs_ptr->intrabc_mode > 0 && config->screen_content_mode != 1) {
        SVT_ERROR(
            "Instance %u: The intra BC feature is only available when screen_content_mode is set "
            "to 1\n",
            channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (scs_ptr->static_config.enable_adaptive_quantization > 2) {
        SVT_ERROR(
            "Instance %u : Invalid enable_adaptive_quantization. enable_adaptive_quantization must "
            "be [0-2]\n",
            channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if ((config->encoder_bit_depth != 8) && (config->encoder_bit_depth != 10)) {
        SVT_ERROR("Instance %u: Encoder Bit Depth shall be only 8 or 10 \n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    // Check if the EncoderBitDepth is conformant with the Profile constraint
    if ((config->profile == 0 || config->profile == 1) && config->encoder_bit_depth > 10) {
        SVT_ERROR(
            "Instance %u: The encoder bit depth shall be equal to 8 or 10 for Main/High Profile\n",
            channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->encoder_color_format != EB_YUV420) {
        SVT_ERROR("Instance %u: Only support 420 now \n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->profile == 0 && config->encoder_color_format > EB_YUV420) {
        SVT_ERROR("Instance %u: Non 420 color format requires profile 1 or 2\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->profile == 1 && config->encoder_color_format != EB_YUV444) {
        SVT_ERROR("Instance %u: Profile 1 requires 4:4:4 color format\n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->profile == 2 && config->encoder_bit_depth <= 10 &&
        config->encoder_color_format != EB_YUV422) {
        SVT_ERROR("Instance %u: Profile 2 bit-depth < 10 requires 4:2:2 color format\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->compressed_ten_bit_format != 0 && config->compressed_ten_bit_format != 1) {
        SVT_ERROR("Instance %u: Invalid Compressed Ten Bit Format flag [0 - 1]\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->use_cpu_flags & CPU_FLAGS_INVALID) {
        SVT_ERROR(
            "Instance %u: param '--asm' have invalid value.\n"
            "Value should be [0 - 11] or [c, mmx, sse, sse2, sse3, ssse3, sse4_1, sse4_2, avx, "
            "avx2, avx512, max]\n",
            channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->target_socket != -1 && config->target_socket != 0 && config->target_socket != 1) {
        SVT_ERROR("Instance %u: Invalid target_socket. target_socket must be [-1 - 1] \n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    // HBD mode decision
    if (scs_ptr->enable_hbd_mode_decision < (int8_t)(-1) || scs_ptr->enable_hbd_mode_decision > 2) {
        SVT_ERROR("Instance %u: Invalid HBD mode decision flag [-1 - 2], your input: %d\n",
                  channel_number + 1,
                  scs_ptr->enable_hbd_mode_decision);
        return_error = EB_ErrorBadParameter;
    }

    // CDEF
    if (config->cdef_level > 4 || config->cdef_level < -1) {
        SVT_ERROR("Instance %u: Invalid CDEF level [0 - 4, -1 for auto], your input: %d\n",
                  channel_number + 1,
                  config->cdef_level);
        return_error = EB_ErrorBadParameter;
    }

    // Restoration Filtering
    if (config->enable_restoration_filtering != 0 && config->enable_restoration_filtering != 1 &&
        config->enable_restoration_filtering != -1) {
        SVT_ERROR("Instance %u: Invalid restoration flag [0 - 1, -1 for auto], your input: %d\n",
                  channel_number + 1,
                  config->enable_restoration_filtering);
        return_error = EB_ErrorBadParameter;
    }

    if (config->enable_mfmv != 0 && config->enable_mfmv != 1 && config->enable_mfmv != -1) {
        SVT_ERROR(
            "Instance %u: Invalid motion field motion vector flag [0/1 or -1 for auto], your "
            "input: %d\n",
            channel_number + 1,
            config->enable_mfmv);
        return_error = EB_ErrorBadParameter;
    }
#if TUNE_FAST_DECODE
    if (config->fast_decode > 4) {
        SVT_ERROR(
            "Instance %u: Invalid fast decode flag [0 - 4, 0 for no decoder optimization], your "
            "input: %d\n",
            channel_number + 1,
            config->fast_decode);
        return_error = EB_ErrorBadParameter;
    }
#else
    if (config->fast_decode > 3) {
        SVT_ERROR(
            "Instance %u: Invalid fast decode flag [0 - 3, 0 for no decoder optimization], your "
            "input: %d\n",
            channel_number + 1,
            config->fast_decode);
        return_error = EB_ErrorBadParameter;
    }
#endif
    if (config->tune > 1) {
        SVT_ERROR(
            "Instance %u: Invalid tune flag [0 - 1, 0 for VQ and 1 for PSNR], your input: %d\n",
            channel_number + 1,
            config->tune);
        return_error = EB_ErrorBadParameter;
    }
    // prediction structure
    if (config->enable_manual_pred_struct) {
        if (config->manual_pred_struct_entry_num > (1 << (MAX_HIERARCHICAL_LEVEL - 1))) {
            SVT_ERROR(
                "Instance %u: Invalid manual prediction structure entry number [1 - 32], your "
                "input: %d\n",
                channel_number + 1,
                config->manual_pred_struct_entry_num);
            return_error = EB_ErrorBadParameter;
        } else {
            for (int32_t i = 0; i < config->manual_pred_struct_entry_num; i++) {
                config->pred_struct[i].ref_list1[REF_LIST_MAX_DEPTH - 1] = 0;
                if (config->pred_struct[i].decode_order >= (1 << (MAX_HIERARCHICAL_LEVEL - 1))) {
                    SVT_ERROR(
                        "Instance %u: Invalid decode order for manual prediction structure [0 - "
                        "31], your input: %d\n",
                        channel_number + 1,
                        config->pred_struct[i].decode_order);
                    return_error = EB_ErrorBadParameter;
                }
                if (config->pred_struct[i].temporal_layer_index >=
                    (1 << (MAX_HIERARCHICAL_LEVEL - 1))) {
                    SVT_ERROR(
                        "Instance %u: Invalid temporal layer index for manual prediction structure "
                        "[0 - 31], your input: %d\n",
                        channel_number + 1,
                        config->pred_struct[i].temporal_layer_index);
                    return_error = EB_ErrorBadParameter;
                }
                Bool  have_ref_frame_within_minigop_in_list0 = FALSE;
                int32_t entry_idx                              = i + 1;
                for (int32_t j = 0; j < REF_LIST_MAX_DEPTH; j++) {
                    if ((entry_idx - config->pred_struct[i].ref_list1[j] >
                         config->manual_pred_struct_entry_num)) {
                        SVT_ERROR(
                            "Instance %u: Invalid ref frame %d in list1 entry%d for manual "
                            "prediction structure, all ref frames in list1 should not exceed "
                            "minigop end\n",
                            channel_number + 1,
                            config->pred_struct[i].ref_list1[j],
                            i);
                        return_error = EB_ErrorBadParameter;
                    }
                    if (config->pred_struct[i].ref_list0[j] < 0) {
                        SVT_ERROR(
                            "Instance %u: Invalid ref frame %d in list0 entry%d for manual "
                            "prediction structure, only forward frames can be in list0\n",
                            channel_number + 1,
                            config->pred_struct[i].ref_list0[j],
                            i);
                        return_error = EB_ErrorBadParameter;
                    }
                    if (!have_ref_frame_within_minigop_in_list0 &&
                        config->pred_struct[i].ref_list0[j] &&
                        entry_idx - config->pred_struct[i].ref_list0[j] >= 0) {
                        have_ref_frame_within_minigop_in_list0 = TRUE;
                    }
                }
                if (!have_ref_frame_within_minigop_in_list0) {
                    SVT_ERROR(
                        "Instance %u: Invalid ref frame in list0 entry%d for manual prediction "
                        "structure,there should be at least one frame within minigop \n",
                        channel_number + 1,
                        i);
                    return_error = EB_ErrorBadParameter;
                }
            }
        }
    }

    if (config->superres_mode > SUPERRES_AUTO) {
        SVT_ERROR("Instance %u: invalid superres-mode %d, should be in the range [%d - %d]\n",
                  channel_number + 1,
                  config->superres_mode,
                  SUPERRES_NONE,
                  SUPERRES_AUTO);
        return_error = EB_ErrorBadParameter;
    }
    if (config->superres_mode > 0 &&
        ((config->rc_stats_buffer.sz || config->pass == ENC_FIRST_PASS))) {
        SVT_ERROR("Instance %u: superres is not supported for 2-pass\n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->superres_qthres > MAX_QP_VALUE) {
        SVT_ERROR("Instance %u: invalid superres-qthres %d, should be in the range [%d - %d] \n",
                  channel_number + 1,
                  config->superres_qthres,
                  MIN_QP_VALUE,
                  MAX_QP_VALUE);
        return_error = EB_ErrorBadParameter;
    }

    if (config->superres_kf_qthres > MAX_QP_VALUE) {
        SVT_ERROR("Instance %u: invalid superres-kf-qthres %d, should be in the range [%d - %d] \n",
                  channel_number + 1,
                  config->superres_kf_qthres,
                  MIN_QP_VALUE,
                  MAX_QP_VALUE);
        return_error = EB_ErrorBadParameter;
    }

    if (config->superres_kf_denom < MIN_SUPERRES_DENOM ||
        config->superres_kf_denom > MAX_SUPERRES_DENOM) {
        SVT_ERROR("Instance %u: invalid superres-kf-denom %d, should be in the range [%d - %d] \n",
                  channel_number + 1,
                  config->superres_kf_denom,
                  MIN_SUPERRES_DENOM,
                  MAX_SUPERRES_DENOM);
        return_error = EB_ErrorBadParameter;
    }

    if (config->superres_denom < MIN_SUPERRES_DENOM ||
        config->superres_denom > MAX_SUPERRES_DENOM) {
        SVT_ERROR("Instance %u: invalid superres-denom %d, should be in the range [%d - %d] \n",
                  channel_number + 1,
                  config->superres_denom,
                  MIN_SUPERRES_DENOM,
                  MAX_SUPERRES_DENOM);
        return_error = EB_ErrorBadParameter;
    }
    if (config->matrix_coefficients == 0 && config->encoder_color_format != EB_YUV444) {
        SVT_ERROR(
            "Instance %u: Identity matrix (matrix_coefficient = 0) may be used only with 4:4:4 "
            "color format.\n",
            channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->hierarchical_levels < 3 || config->hierarchical_levels > 5) {
        SVT_ERROR("Instance %u: Only hierarchical levels 3-5 is currently supported.\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
#if FTR_CBR
    if (config->rate_control_mode == 1 && config->intra_period_length == -1) {
#else
    if (config->rate_control_mode != 0 && config->intra_period_length == -1) {
#endif
        SVT_ERROR(
            "Instance %u: keyint = -1 is not supported for modes other than CRF rate control "
            "encoding modes.\n",
            channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    // Limit 8K & 16K configurations ( due to  memory constraints)
    if ((uint64_t)(scs_ptr->max_input_luma_width * scs_ptr->max_input_luma_height) >
            INPUT_SIZE_4K_TH &&
        config->enc_mode <= ENC_M7) {
        SVT_ERROR("Instance %u: 8k+ resolution support is limited to M8 and faster presets.\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->enable_adaptive_quantization == 1 &&
        (config->tile_columns > 0 || config->tile_rows > 0)) {
        SVT_ERROR(
            "Instance %u: Adaptive quantization using segmentation is not supported in combination "
            "with tiles.\n",
            channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (config->pass > 0 && scs_ptr->static_config.enable_overlays) {
        SVT_ERROR(
            "Instance %u: The overlay frames feature is currently not supported with multi-pass "
            "encoding\n",
            channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
    int pass = config->pass;

    if (pass != 3 && pass != 2 && pass != 1 && pass != 0) {
        SVT_ERROR("Instance %u: %d pass encode is not supported. --pass has a range of [0-3]\n",
                  channel_number + 1,
                  pass);
        return_error = EB_ErrorBadParameter;
    }

    if (config->intra_refresh_type != 2 && pass > 0) {
        SVT_ERROR("Instance %u: Multi-pass encode only supports closed-gop configurations.\n",
                  channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

    if (pass == 3 && config->rate_control_mode == 0) {
        SVT_ERROR("Instance %u: CRF does not support 3-pass\n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }

#if FIX_AQ_MODE
    if (config->enable_adaptive_quantization == 0 && config->rate_control_mode) {
        SVT_ERROR("Instance %u: Adaptive quantization can not be turned OFF when RC ON\n", channel_number + 1);
        return_error = EB_ErrorBadParameter;
    }
#endif
    /* Warnings about the use of features that are incomplete */

    // color description
    if (config->color_primaries == 0 || config->color_primaries == 3 ||
        (config->color_primaries >= 13 && config->color_primaries <= 21) ||
        config->color_primaries > 22) {
        SVT_WARN(
            "Instance %u: value %u for color_primaries is reserved and not recommended for "
            "usage.\n",
            channel_number + 1,
            config->color_primaries);
    }
    if (config->transfer_characteristics == 0 || config->transfer_characteristics == 3 ||
        config->transfer_characteristics > 18) {
        SVT_WARN(
            "Instance %u: value %u for transfer_characteristics is reserved and not recommended "
            "for usage.\n",
            channel_number + 1,
            config->transfer_characteristics);
    }

    if (config->matrix_coefficients == 3 || config->matrix_coefficients > 14) {
        SVT_WARN(
            "Instance %u: value %u for matrix_coefficients is reserved and not recommended for "
            "usage.\n",
            channel_number + 1,
            config->matrix_coefficients);
    }

    if (config->rate_control_mode == 1 || config->rate_control_mode == 2) {
        SVT_WARN(
            "Instance %u: The VBR and CBR rate control modes are a work-in-progress projects, and "
            "are only available for demos, experimentation, and further development uses and "
            "should not be used for benchmarking until fully implemented.\n",
            channel_number + 1);
    }

    if (config->film_grain_denoise_strength > 0 && config->enc_mode > 3) {
        SVT_WARN(
            "Instance %u: It is recommended to not use Film Grain for presets greater than 3 as it "
            "produces a significant compute overhead. This combination should only be used for "
            "debug purposes.\n",
            channel_number + 1);
    }

    // Limit 8K & 16K support
    if ((uint64_t)(scs_ptr->max_input_luma_width * scs_ptr->max_input_luma_height) >
        INPUT_SIZE_4K_TH) {
        SVT_WARN(
            "Instance %u: 8K and higher resolution support is currently a work-in-progress "
            "project, and is only available for demos, experimentation, and further development "
            "uses and should not be used for benchmarking until fully implemented.\n",
            channel_number + 1);
    }

    if (config->pred_structure == 1) {
        SVT_WARN(
            "Instance %u: The low delay encoding mode is a work-in-progress project, and is only "
            "available for demos, experimentation, and further development uses and should not be "
            "used for benchmarking until fully implemented.\n",
            channel_number + 1);
        if (config->tune == 0) {
            SVT_WARN(
                "Instance %u: Tune 0 is not applicable for low-delay, tune will be forced to 1.\n",
                channel_number + 1);
            config->tune = 1;
        }

        if (config->superres_mode != 0) {
            SVT_ERROR("Instance %u: Superres is not supported for low-delay.\n",
                      channel_number + 1);
            return_error = EB_ErrorBadParameter;
        }

        if (config->enable_overlays) {
            SVT_ERROR("Instance %u: Overlay is not supported for low-delay.\n", channel_number + 1);
            return_error = EB_ErrorBadParameter;
        }
    }

    return return_error;
}

/**********************************
Set Default Library Params
**********************************/
EbErrorType svt_av1_set_default_params(EbSvtAv1EncConfiguration *config_ptr) {
    EbErrorType return_error = EB_ErrorNone;

    if (!config_ptr) {
        SVT_ERROR("The EbSvtAv1EncConfiguration structure is empty!\n");
        return EB_ErrorBadParameter;
    }

    config_ptr->frame_rate                = 60 << 16;
    config_ptr->frame_rate_numerator      = 60000;
    config_ptr->frame_rate_denominator    = 1000;
    config_ptr->encoder_bit_depth         = 8;
    config_ptr->compressed_ten_bit_format = 0;
    config_ptr->source_width              = 0;
    config_ptr->source_height             = 0;
    config_ptr->stat_report               = 0;
    config_ptr->tile_rows                 = 0;
    config_ptr->tile_columns              = 0;
#if FIX_1PVBR
    config_ptr->qp = DEFAULT_QP;
#else
    config_ptr->qp = 50;
#endif
    config_ptr->use_qp_file = FALSE;

    config_ptr->use_fixed_qindex_offsets = FALSE;
    memset(config_ptr->qindex_offsets, 0, sizeof(config_ptr->qindex_offsets));
    config_ptr->key_frame_chroma_qindex_offset = 0;
    config_ptr->key_frame_qindex_offset        = 0;
    memset(config_ptr->chroma_qindex_offsets, 0, sizeof(config_ptr->chroma_qindex_offsets));
    config_ptr->luma_y_dc_qindex_offset = 0;
    config_ptr->chroma_u_dc_qindex_offset = 0;
    config_ptr->chroma_u_ac_qindex_offset = 0;
    config_ptr->chroma_v_dc_qindex_offset = 0;
    config_ptr->chroma_v_ac_qindex_offset = 0;

    config_ptr->scene_change_detection = 0;
    config_ptr->rate_control_mode      = 0;
    config_ptr->look_ahead_distance    = (uint32_t)~0;
    config_ptr->enable_tpl_la          = 1;
    config_ptr->target_bit_rate        = 2000000;
    config_ptr->max_bit_rate           = 0;
    config_ptr->max_qp_allowed         = 63;
    config_ptr->min_qp_allowed         = 1;

    config_ptr->enable_adaptive_quantization = 2;
    config_ptr->enc_mode                     = 12;
    config_ptr->intra_period_length          = -2;
    config_ptr->intra_refresh_type           = 2;
    config_ptr->hierarchical_levels          = 4;
    config_ptr->pred_structure               = PRED_RANDOM_ACCESS;
    config_ptr->enable_dlf_flag              = TRUE;
    config_ptr->cdef_level                   = DEFAULT;
    config_ptr->enable_restoration_filtering = DEFAULT;
    config_ptr->enable_mfmv                  = DEFAULT;
    config_ptr->fast_decode                  = 0;
    memset(config_ptr->pred_struct, 0, sizeof(config_ptr->pred_struct));
    config_ptr->enable_manual_pred_struct    = FALSE;
    config_ptr->manual_pred_struct_entry_num = 0;
    config_ptr->encoder_color_format         = EB_YUV420;
    // Two pass data rate control options
    config_ptr->vbr_bias_pct             = 50;
    config_ptr->vbr_min_section_pct      = 0;
    config_ptr->vbr_max_section_pct      = 2000;
    config_ptr->under_shoot_pct          = 25;
    config_ptr->over_shoot_pct           = 25;
    config_ptr->mbr_over_shoot_pct       = 50;
    config_ptr->maximum_buffer_size_ms   = 6000;
    config_ptr->starting_buffer_level_ms = 4000;
    config_ptr->optimal_buffer_level_ms  = 5000;
    config_ptr->recode_loop              = ALLOW_RECODE_DEFAULT;
    // Bitstream options
    //config_ptr->codeVpsSpsPps = 0;
    //config_ptr->codeEosNal = 0;
    config_ptr->restricted_motion_vector = FALSE;

    config_ptr->high_dynamic_range_input = 0;
    config_ptr->screen_content_mode      = 2;

    // Annex A parameters
    config_ptr->profile = 0;
    config_ptr->tier    = 0;
    config_ptr->level   = 0;

    // Latency
    config_ptr->film_grain_denoise_strength = 0;

    // CPU Flags
    config_ptr->use_cpu_flags = CPU_FLAGS_ALL;

    // Channel info
    config_ptr->logical_processors   = 0;
    config_ptr->pin_threads          = 0;
    config_ptr->target_socket        = -1;
    config_ptr->channel_id           = 0;
    config_ptr->active_channel_count = 1;

    // Debug info
    config_ptr->recon_enabled = 0;

    // Alt-Ref default values
    config_ptr->enable_tf       = TRUE;
    config_ptr->enable_overlays = FALSE;
    config_ptr->tune            = 1;
    // Super-resolution default values
    config_ptr->superres_mode      = SUPERRES_NONE;
    config_ptr->superres_denom     = 8;
    config_ptr->superres_kf_denom  = 8;
    config_ptr->superres_qthres    = 43; // random threshold, change
    config_ptr->superres_kf_qthres = 43; // random threshold, change

    // Color description default values
    config_ptr->color_description_present_flag = FALSE;
    config_ptr->color_primaries                = 2;
    config_ptr->transfer_characteristics       = 2;
    config_ptr->matrix_coefficients            = 2;
    config_ptr->color_range                    = 0;
    config_ptr->pass                           = 0;
    memset(&config_ptr->mastering_display, 0, sizeof(config_ptr->mastering_display));
    memset(&config_ptr->content_light_level, 0, sizeof(config_ptr->content_light_level));
    return return_error;
}

static const char *tier_to_str(unsigned in) {
    if (!in)
        return "(auto)";
    static char ret[11];
    snprintf(ret, 11, "%u", in);
    return ret;
}
static const char *level_to_str(unsigned in) {
    if (!in)
        return "(auto)";
    static char ret[313];
    snprintf(ret, 313, "%.1f", in / 10.0);
    return ret;
}

//#define DEBUG_BUFFERS
void svt_av1_print_lib_params(SequenceControlSet *scs) {
    EbSvtAv1EncConfiguration *config = &scs->static_config;

    SVT_INFO("-------------------------------------------\n");

    if (config->pass == ENC_FIRST_PASS)
        SVT_INFO("SVT [config]: Preset \t\t\t\t\t\t\t: Pass 1\n");
    else if (config->pass == ENC_MIDDLE_PASS)
        SVT_INFO("SVT [config]: Preset \t\t\t\t\t\t\t: Pass 2\n");
    else {
        SVT_INFO("SVT [config]: %s\tTier %s\tLevel %s\n",
                 config->profile == MAIN_PROFILE               ? "Main Profile"
                     : config->profile == HIGH_PROFILE         ? "High Profile"
                     : config->profile == PROFESSIONAL_PROFILE ? "Professional Profile"
                                                               : "Unknown Profile",
                 tier_to_str(config->tier),
                 level_to_str(config->level));
        SVT_INFO("SVT [config]: Preset \t\t\t\t\t\t\t: %d\n", config->enc_mode);
        SVT_INFO(
            "SVT [config]: EncoderBitDepth / EncoderColorFormat / CompressedTenBitFormat\t: %d / "
            "%d / %d\n",
            config->encoder_bit_depth,
            config->encoder_color_format,
            config->compressed_ten_bit_format);
        SVT_INFO("SVT [config]: SourceWidth / SourceHeight\t\t\t\t\t: %d / %d\n",
                 config->source_width,
                 config->source_height);
        if (config->frame_rate_denominator != 0 && config->frame_rate_numerator != 0)
            SVT_INFO(
                "SVT [config]: Fps_Numerator / Fps_Denominator / Gop Size / IntraRefreshType \t: "
                "%d / %d / %d / %d\n",
                config->frame_rate_numerator,
                config->frame_rate_denominator,
                config->intra_period_length + 1,
                config->intra_refresh_type);
        else
            SVT_INFO("SVT [config]: FrameRate / Gop Size\t\t\t\t\t\t: %d / %d\n",
                     config->frame_rate > 1000 ? config->frame_rate >> 16 : config->frame_rate,
                     config->intra_period_length + 1);
        SVT_INFO("SVT [config]: HierarchicalLevels  / PredStructure\t\t\t\t: %d / %d\n",
                 config->hierarchical_levels,
                 config->pred_structure);
        switch (config->rate_control_mode) {
        case 0:
            if (config->max_bit_rate)
                SVT_INFO(
                    "SVT [config]: BRC Mode / %s / MaxBitrate (kbps)/ SceneChange\t\t: %s / %d / "
                    "%d / %d\n",
                    scs->tpl_level ? "Rate Factor" : "CQP Assignment",
                    scs->tpl_level ? "Capped CRF" : "CQP",
                    scs->static_config.qp,
                    (int)config->max_bit_rate / 1000,
                    config->scene_change_detection);
#if FIX_AQ_MODE
            else if (scs->tpl_level)
                SVT_INFO("SVT [config]: BRC Mode / %s / SceneChange\t\t\t\t: %s / %d / %d\n",
                    "Rate Factor",
                    "CRF",
                    scs->static_config.qp,
                    config->scene_change_detection);
            else
                SVT_INFO("SVT [config]: BRC Mode / %s / SceneChange\t\t\t: %s / %d / %d\n",
                    "CQP Assignment",
                    "CQP",
                    scs->static_config.qp,
                    config->scene_change_detection);
#else
            else
                SVT_INFO("SVT [config]: BRC Mode / %s / SceneChange\t\t\t\t: %s / %d / %d\n",
                         scs->tpl_level ? "Rate Factor" : "CQP Assignment",
                         scs->tpl_level ? "CRF" : "CQP",
                         scs->static_config.qp,
                         config->scene_change_detection);
#endif
            break;
        case 1:
            SVT_INFO(
                "SVT [config]: BRC Mode / TargetBitrate (kbps)/ SceneChange\t\t\t: VBR / %d / %d\n",
                (int)config->target_bit_rate / 1000,
                config->scene_change_detection);
            break;
        case 2:
#if FTR_CBR
            SVT_INFO(
                "SVT [config]: BRC Mode / TargetBitrate (kbps)/ SceneChange\t\t\t: CBR "
                "/ %d / %d\n",
                (int)config->target_bit_rate / 1000,
                config->scene_change_detection);
#else
            SVT_INFO(
                "SVT [config]: BRC Mode / TargetBitrate (kbps)/ SceneChange\t\t\t: Constraint VBR "
                "/ %d / %d\n",
                (int)config->target_bit_rate / 1000,
                config->scene_change_detection);
#endif
            break;
        }
    }
#ifdef DEBUG_BUFFERS
    SVT_INFO("SVT [config]: INPUT / OUTPUT \t\t\t\t\t\t\t: %d / %d\n",
             scs->input_buffer_fifo_init_count,
             scs->output_stream_buffer_fifo_init_count);
    SVT_INFO("SVT [config]: CPCS / PAREF / REF / ME\t\t\t\t\t\t: %d / %d / %d / %d\n",
             scs->picture_control_set_pool_init_count_child,
             scs->pa_reference_picture_buffer_init_count,
             scs->reference_picture_buffer_init_count,
             scs->me_pool_init_count);
    SVT_INFO(
        "SVT [config]: ME_SEG_W0 / ME_SEG_W1 / ME_SEG_W2 / ME_SEG_W3 \t\t\t: %d / %d / %d / %d\n",
        scs->me_segment_column_count_array[0],
        scs->me_segment_column_count_array[1],
        scs->me_segment_column_count_array[2],
        scs->me_segment_column_count_array[3]);
    SVT_INFO(
        "SVT [config]: ME_SEG_H0 / ME_SEG_H1 / ME_SEG_H2 / ME_SEG_H3 \t\t\t: %d / %d / %d / %d\n",
        scs->me_segment_row_count_array[0],
        scs->me_segment_row_count_array[1],
        scs->me_segment_row_count_array[2],
        scs->me_segment_row_count_array[3]);
    SVT_INFO(
        "SVT [config]: ME_SEG_W0 / ME_SEG_W1 / ME_SEG_W2 / ME_SEG_W3 \t\t\t: %d / %d / %d / %d\n",
        scs->enc_dec_segment_col_count_array[0],
        scs->enc_dec_segment_col_count_array[1],
        scs->enc_dec_segment_col_count_array[2],
        scs->enc_dec_segment_col_count_array[3]);
    SVT_INFO(
        "SVT [config]: ME_SEG_H0 / ME_SEG_H1 / ME_SEG_H2 / ME_SEG_H3 \t\t\t: %d / %d / %d / %d\n",
        scs->enc_dec_segment_row_count_array[0],
        scs->enc_dec_segment_row_count_array[1],
        scs->enc_dec_segment_row_count_array[2],
        scs->enc_dec_segment_row_count_array[3]);
    SVT_INFO(
        "SVT [config]: PA_P / ME_P / SBO_P / MDC_P / ED_P / EC_P \t\t\t: %d / %d / %d / %d / %d / "
        "%d\n",
        scs->picture_analysis_process_init_count,
        scs->motion_estimation_process_init_count,
        scs->source_based_operations_process_init_count,
        scs->mode_decision_configuration_process_init_count,
        scs->enc_dec_process_init_count,
        scs->entropy_coding_process_init_count);
    SVT_INFO("SVT [config]: DLF_P / CDEF_P / REST_P \t\t\t\t\t\t: %d / %d / %d\n",
             scs->dlf_process_init_count,
             scs->cdef_process_init_count,
             scs->rest_process_init_count);
#endif
    SVT_INFO("-------------------------------------------\n");

    fflush(stdout);
}

/**********************************
* Parse Single Parameter
**********************************/
//assume the input list of values are in the format of "[v1,v2,v3,...]"
static EbErrorType parse_list(const char *nptr, int32_t *list, size_t n) {
    const char *ptr = nptr;
    char       *endptr;
    size_t      i = 0;
    while (*ptr) {
        if (*ptr == '[' || *ptr == ']') {
            ptr++;
            continue;
        }

        int32_t rawval = strtol(ptr, &endptr, 10);
        if (i >= n) {
            return EB_ErrorBadParameter;
        } else if (*endptr == ',' || *endptr == ']') {
            endptr++;
        } else if (*endptr) {
            return EB_ErrorBadParameter;
        }
        list[i++] = rawval;
        ptr       = endptr;
    }
    return EB_ErrorNone;
}

static EbErrorType str_to_int64(const char *nptr, int64_t *out) {
    char   *endptr;
    int64_t val;

    val = strtoll(nptr, &endptr, 0);

    if (endptr == nptr || *endptr)
        return EB_ErrorBadParameter;

    *out = val;
    return EB_ErrorNone;
}

static EbErrorType str_to_int(const char *nptr, int32_t *out) {
    char   *endptr;
    int32_t val;

    val = strtol(nptr, &endptr, 0);

    if (endptr == nptr || *endptr)
        return EB_ErrorBadParameter;

    *out = val;
    return EB_ErrorNone;
}

static EbErrorType str_to_uint(const char *nptr, uint32_t *out) {
    char    *endptr;
    uint32_t val;

    val = strtoul(nptr, &endptr, 0);

    if (endptr == nptr || *endptr)
        return EB_ErrorBadParameter;

    *out = val;
    return EB_ErrorNone;
}

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif

static EbErrorType str_to_bool(const char *nptr, Bool *out) {
    Bool val;

    if (!strcmp(nptr, "1") || !strcasecmp(nptr, "true") || !strcasecmp(nptr, "yes"))
        val = TRUE;
    else if (!strcmp(nptr, "0") || !strcasecmp(nptr, "false") || !strcasecmp(nptr, "no"))
        val = FALSE;
    else
        return EB_ErrorBadParameter;

    *out = val;
    return EB_ErrorNone;
}

static EbErrorType str_to_crf(const char *nptr, EbSvtAv1EncConfiguration *config_struct) {
    uint32_t    crf;
    EbErrorType return_error;

    return_error = str_to_uint(nptr, &crf);
    if (return_error == EB_ErrorBadParameter)
        return return_error;

    config_struct->qp                = crf;
    config_struct->rate_control_mode = 0;
    config_struct->enable_tpl_la     = 1;

    return EB_ErrorNone;
}

static EbErrorType str_to_keyint(const char *nptr, int32_t *out) {
    int32_t     keyint;
    EbErrorType return_error;

    return_error = str_to_int(nptr, &keyint);
    if (return_error == EB_ErrorBadParameter)
        return return_error;

    *out = keyint < 0 ? keyint : keyint - 1;

    return EB_ErrorNone;
}

static EbErrorType str_to_profile(const char *nptr, EbAv1SeqProfile *out) {
    const struct {
        const char     *name;
        EbAv1SeqProfile profile;
    } profiles[] = {
        {"main", MAIN_PROFILE},
        {"high", HIGH_PROFILE},
        {"professional", PROFESSIONAL_PROFILE},
    };
    const size_t profiles_size = sizeof(profiles) / sizeof(profiles[0]);

    for (size_t i = 0; i < profiles_size; i++) {
        if (!strcmp(nptr, profiles[i].name)) {
            *out = profiles[i].profile;
            return EB_ErrorNone;
        }
    }

    return EB_ErrorBadParameter;
}

static EbErrorType str_to_color_fmt(const char *nptr, EbColorFormat *out) {
    const struct {
        const char   *name;
        EbColorFormat fmt;
    } color_formats[] = {
        {"mono", EB_YUV400},
        {"400", EB_YUV400},
        {"420", EB_YUV420},
        {"422", EB_YUV422},
        {"444", EB_YUV444},
    };
    const size_t color_format_size = sizeof(color_formats) / sizeof(color_formats[0]);

    for (size_t i = 0; i < color_format_size; i++) {
        if (!strcmp(nptr, color_formats[i].name)) {
            *out = color_formats[i].fmt;
            return EB_ErrorNone;
        }
    }

    return EB_ErrorBadParameter;
}

static EbErrorType str_to_intra_rt(const char *nptr, SvtAv1IntraRefreshType *out) {
    const struct {
        const char            *name;
        SvtAv1IntraRefreshType type;
    } refresh_types[] = {
        {"cra", SVT_AV1_FWDKF_REFRESH},
        {"fwdkf", SVT_AV1_FWDKF_REFRESH},
        {"idr", SVT_AV1_KF_REFRESH},
        {"kf", SVT_AV1_KF_REFRESH},
    };
    const size_t refresh_type_size = sizeof(refresh_types) / sizeof(refresh_types[0]);

    for (size_t i = 0; i < refresh_type_size; i++) {
        if (!strcmp(nptr, refresh_types[i].name)) {
            *out = refresh_types[i].type;
            return EB_ErrorNone;
        }
    }

    return EB_ErrorBadParameter;
}

static EbErrorType str_to_color_primaries(const char *nptr, uint8_t *out) {
    const struct {
        const char      *name;
        EbColorPrimaries primaries;
    } color_primaries[] = {
        {"bt709", EB_CICP_CP_BT_709},
        {"bt470m", EB_CICP_CP_BT_470_M},
        {"bt470bg", EB_CICP_CP_BT_470_B_G},
        {"bt601", EB_CICP_CP_BT_601},
        {"smpte240", EB_CICP_CP_SMPTE_240},
        {"film", EB_CICP_CP_GENERIC_FILM},
        {"bt2020", EB_CICP_CP_BT_2020},
        {"xyz", EB_CICP_CP_XYZ},
        {"smpte431", EB_CICP_CP_SMPTE_431},
        {"smpte432", EB_CICP_CP_SMPTE_432},
        {"ebu3213", EB_CICP_CP_EBU_3213},
    };
    const size_t color_primaries_size = sizeof(color_primaries) / sizeof(color_primaries[0]);

    for (size_t i = 0; i < color_primaries_size; i++) {
        if (!strcmp(nptr, color_primaries[i].name)) {
            *out = color_primaries[i].primaries;
            return EB_ErrorNone;
        }
    }

    return EB_ErrorBadParameter;
}

static EbErrorType str_to_transfer_characteristics(const char *nptr, uint8_t *out) {
    const struct {
        const char               *name;
        EbTransferCharacteristics tfc;
    } transfer_characteristics[] = {
        {"bt709", EB_CICP_TC_BT_709},
        {"bt470m", EB_CICP_TC_BT_470_M},
        {"bt470bg", EB_CICP_TC_BT_470_B_G},
        {"bt601", EB_CICP_TC_BT_601},
        {"smpte240", EB_CICP_TC_SMPTE_240},
        {"linear", EB_CICP_TC_LINEAR},
        {"log100", EB_CICP_TC_LOG_100},
        {"log100-sqrt10", EB_CICP_TC_LOG_100_SQRT10},
        {"iec61966", EB_CICP_TC_IEC_61966},
        {"bt1361", EB_CICP_TC_BT_1361},
        {"srgb", EB_CICP_TC_SRGB},
        {"bt2020-10", EB_CICP_TC_BT_2020_10_BIT},
        {"bt2020-12", EB_CICP_TC_BT_2020_12_BIT},
        {"smpte2084", EB_CICP_TC_SMPTE_2084},
        {"smpte428", EB_CICP_TC_SMPTE_428},
        {"hlg", EB_CICP_TC_HLG},
    };
    const size_t transfer_characteristics_size = sizeof(transfer_characteristics) /
        sizeof(transfer_characteristics[0]);

    for (size_t i = 0; i < transfer_characteristics_size; i++) {
        if (!strcmp(nptr, transfer_characteristics[i].name)) {
            *out = transfer_characteristics[i].tfc;
            return EB_ErrorNone;
        }
    }

    return EB_ErrorBadParameter;
}

static EbErrorType str_to_matrix_coefficients(const char *nptr, uint8_t *out) {
    const struct {
        const char          *name;
        EbMatrixCoefficients coeff;
    } matrix_coefficients[] = {
        {"identity", EB_CICP_MC_IDENTITY},
        {"bt709", EB_CICP_MC_BT_709},
        {"fcc", EB_CICP_MC_FCC},
        {"bt470bg", EB_CICP_MC_BT_470_B_G},
        {"bt601", EB_CICP_MC_BT_601},
        {"smpte240", EB_CICP_MC_SMPTE_240},
        {"ycgco", EB_CICP_MC_SMPTE_YCGCO},
        {"bt2020-ncl", EB_CICP_MC_BT_2020_NCL},
        {"bt2020-cl", EB_CICP_MC_BT_2020_CL},
        {"smpte2085", EB_CICP_MC_SMPTE_2085},
        {"chroma-ncl", EB_CICP_MC_CHROMAT_NCL},
        {"chroma-cl", EB_CICP_MC_CHROMAT_CL},
        {"ictcp", EB_CICP_MC_ICTCP},
    };
    const size_t matrix_coefficients_size = sizeof(matrix_coefficients) /
        sizeof(matrix_coefficients[0]);

    for (size_t i = 0; i < matrix_coefficients_size; i++) {
        if (!strcmp(nptr, matrix_coefficients[i].name)) {
            *out = matrix_coefficients[i].coeff;
            return EB_ErrorNone;
        }
    }

    return EB_ErrorBadParameter;
}

static EbErrorType str_to_color_range(const char *nptr, uint8_t *out) {
    const struct {
        const char  *name;
        EbColorRange range;
    } color_range[] = {
        {"studio", EB_CR_STUDIO_RANGE},
        {"full", EB_CR_FULL_RANGE},
    };
    const size_t color_range_size = sizeof(color_range) / sizeof(color_range[0]);

    for (size_t i = 0; i < color_range_size; i++) {
        if (!strcmp(nptr, color_range[i].name)) {
            *out = color_range[i].range;
            return EB_ErrorNone;
        }
    }

    return EB_ErrorBadParameter;
}

#define COLOR_OPT(par, opt)                                          \
    do {                                                             \
        if (!strcmp(name, par)) {                                    \
            return_error = str_to_##opt(value, &config_struct->opt); \
            if (return_error == EB_ErrorNone)                        \
                return return_error;                                 \
            uint32_t val;                                            \
            return_error = str_to_uint(value, &val);                 \
            if (return_error == EB_ErrorNone)                        \
                config_struct->opt = val;                            \
            return return_error;                                     \
        }                                                            \
    } while (0)

#define COLOR_METADATA_OPT(par, opt)                                                       \
    do {                                                                                   \
        if (!strcmp(name, par))                                                            \
            return svt_aom_parse_##opt(&config_struct->opt, value) ? EB_ErrorNone          \
                                                                   : EB_ErrorBadParameter; \
    } while (0)

EB_API EbErrorType svt_av1_enc_parse_parameter(EbSvtAv1EncConfiguration *config_struct,
                                               const char *name, const char *value) {
    if (config_struct == NULL || name == NULL || value == NULL)
        return EB_ErrorBadParameter;

    EbErrorType return_error = EB_ErrorBadParameter;

    if (!strcmp(name, "keyint"))
        return str_to_keyint(value, &config_struct->intra_period_length);

    // options updating more than one field
    if (!strcmp(name, "crf"))
        return str_to_crf(value, config_struct);

    // custom enum fields
    if (!strcmp(name, "profile"))
        return str_to_profile(value, &config_struct->profile) == EB_ErrorBadParameter
            ? str_to_uint(value, (uint32_t *)&config_struct->profile)
            : EB_ErrorNone;

    if (!strcmp(name, "color-format"))
        return str_to_color_fmt(value, &config_struct->encoder_color_format) == EB_ErrorBadParameter
            ? str_to_uint(value, (uint32_t *)&config_struct->encoder_color_format)
            : EB_ErrorNone;

    if (!strcmp(name, "irefresh-type"))
        return str_to_intra_rt(value, &config_struct->intra_refresh_type) == EB_ErrorBadParameter
            ? str_to_uint(value, (uint32_t *)&config_struct->intra_refresh_type)
            : EB_ErrorNone;

    COLOR_OPT("color-primaries", color_primaries);
    COLOR_OPT("transfer-characteristics", transfer_characteristics);
    COLOR_OPT("matrix-coefficients", matrix_coefficients);
    COLOR_OPT("color-range", color_range);

    // custom struct fields
    COLOR_METADATA_OPT("mastering-display", mastering_display);
    COLOR_METADATA_OPT("content-light", content_light_level);

    // arrays
    if (!strcmp(name, "qindex-offsets"))
        return parse_list(value, config_struct->qindex_offsets, EB_MAX_TEMPORAL_LAYERS);

    if (!strcmp(name, "chroma-qindex-offsets"))
        return parse_list(value, config_struct->chroma_qindex_offsets, EB_MAX_TEMPORAL_LAYERS);

    // uint32_t fields
    const struct {
        const char *name;
        uint32_t   *out;
    } uint_opts[] = {
        {"width", &config_struct->source_width},
        {"height", &config_struct->source_height},
        {"qp", &config_struct->qp},
        {"film-grain", &config_struct->film_grain_denoise_strength},
        {"hierarchical-levels", &config_struct->hierarchical_levels},
        {"tier", &config_struct->tier},
        {"level", &config_struct->level},
        {"lp", &config_struct->logical_processors},
        {"pin", &config_struct->pin_threads},
        {"fps-num", &config_struct->frame_rate_numerator},
        {"fps-denom", &config_struct->frame_rate_denominator},
        {"rc", &config_struct->rate_control_mode},
        {"lookahead", &config_struct->look_ahead_distance},
        {"tbr", &config_struct->target_bit_rate},
        {"mbr", &config_struct->max_bit_rate},
#if !FTR_CBR
        {"vbv-bufsize", &config_struct->vbv_bufsize},
#endif
        {"scd", &config_struct->scene_change_detection},
        {"max-qp", &config_struct->max_qp_allowed},
        {"min-qp", &config_struct->min_qp_allowed},
        {"bias-pct", &config_struct->vbr_bias_pct},
        {"minsection-pct", &config_struct->vbr_min_section_pct},
        {"maxsection-pct", &config_struct->vbr_max_section_pct},
        {"undershoot-pct", &config_struct->under_shoot_pct},
        {"overshoot-pct", &config_struct->over_shoot_pct},
        {"recode-loop", &config_struct->recode_loop},
        {"enable-stat-report", &config_struct->stat_report},
        {"scm", &config_struct->screen_content_mode},
        {"input-depth", &config_struct->encoder_bit_depth},
        {"compressed-ten-bit-format", &config_struct->compressed_ten_bit_format},
    };
    const size_t uint_opts_size = sizeof(uint_opts) / sizeof(uint_opts[0]);

    for (size_t i = 0; i < uint_opts_size; i++) {
        if (!strcmp(name, uint_opts[i].name)) {
            return str_to_uint(value, uint_opts[i].out);
        }
    }

    // uint8_t fields
    const struct {
        const char *name;
        uint8_t    *out;
    } uint8_opts[] = {
        {"pred-struct", &config_struct->pred_structure},
        {"enable-tpl-la", &config_struct->enable_tpl_la},
        {"aq-mode", &config_struct->enable_adaptive_quantization},
        {"superres-mode", &config_struct->superres_mode},
        {"superres-qthres", &config_struct->superres_qthres},
        {"superres-kf-qthres", &config_struct->superres_kf_qthres},
        {"superres-denom", &config_struct->superres_denom},
        {"superres-kf-denom", &config_struct->superres_kf_denom},
        {"fast-decode", &config_struct->fast_decode},
        {"tune", &config_struct->tune},
    };
    const size_t uint8_opts_size = sizeof(uint8_opts) / sizeof(uint8_opts[0]);

    for (size_t i = 0; i < uint8_opts_size; i++) {
        if (!strcmp(name, uint8_opts[i].name)) {
            uint32_t val;
            return_error = str_to_uint(value, &val);
            if (return_error == EB_ErrorNone)
                *uint8_opts[i].out = val;
            return return_error;
        }
    }

    // int64_t fields
    const struct {
        const char *name;
        int64_t    *out;
    } int64_opts[] = {
        {"buf-initial-sz", &config_struct->starting_buffer_level_ms},
        {"buf-optimal-sz", &config_struct->optimal_buffer_level_ms},
        {"buf-sz", &config_struct->maximum_buffer_size_ms},
    };
    const size_t int64_opts_size = sizeof(int64_opts) / sizeof(int64_opts[0]);

    for (size_t i = 0; i < int64_opts_size; i++) {
        if (!strcmp(name, int64_opts[i].name)) {
            return str_to_int64(value, int64_opts[i].out);
        }
    }

    // int32_t fields
    const struct {
        const char *name;
        int32_t    *out;
    } int_opts[] = {
        {"key-frame-chroma-qindex-offset", &config_struct->key_frame_chroma_qindex_offset},
        {"key-frame-qindex-offset", &config_struct->key_frame_qindex_offset},
        {"luma-y-dc-qindex-offset", &config_struct->luma_y_dc_qindex_offset},
        {"chroma-u-dc-qindex-offset", &config_struct->chroma_u_dc_qindex_offset},
        {"chroma-u-ac-qindex-offset", &config_struct->chroma_u_ac_qindex_offset},
        {"chroma-v-dc-qindex-offset", &config_struct->chroma_v_dc_qindex_offset},
        {"chroma-v-ac-qindex-offset", &config_struct->chroma_v_ac_qindex_offset},
        {"pass", &config_struct->pass},
        {"enable-cdef", &config_struct->cdef_level},
        {"enable-restoration", &config_struct->enable_restoration_filtering},
        {"enable-mfmv", &config_struct->enable_mfmv},
        {"intra-period", &config_struct->intra_period_length},
        {"tile-rows", &config_struct->tile_rows},
        {"tile-columns", &config_struct->tile_columns},
        {"ss", &config_struct->target_socket},
    };
    const size_t int_opts_size = sizeof(int_opts) / sizeof(int_opts[0]);

    for (size_t i = 0; i < int_opts_size; i++) {
        if (!strcmp(name, int_opts[i].name)) {
            return str_to_int(value, int_opts[i].out);
        }
    }

    // int8_t fields
    const struct {
        const char *name;
        int8_t     *out;
    } int8_opts[] = {
        {"preset", &config_struct->enc_mode},
    };
    const size_t int8_opts_size = sizeof(int8_opts) / sizeof(int8_opts[0]);

    for (size_t i = 0; i < int8_opts_size; i++) {
        if (!strcmp(name, int8_opts[i].name)) {
            int32_t val;
            return_error = str_to_int(value, &val);
            if (return_error == EB_ErrorNone)
                *int8_opts[i].out = val;
            return return_error;
        }
    }

    // Bool fields
    const struct {
        const char *name;
        Bool     *out;
    } bool_opts[] = {
        {"use-q-file", &config_struct->use_qp_file},
        {"use-fixed-qindex-offsets", &config_struct->use_fixed_qindex_offsets},
        {"enable-dlf", &config_struct->enable_dlf_flag},
        {"rmv", &config_struct->restricted_motion_vector},
        {"enable-tf", &config_struct->enable_tf},
        {"enable-overlays", &config_struct->enable_overlays},
        {"enable-hdr", &config_struct->high_dynamic_range_input},
    };
    const size_t bool_opts_size = sizeof(bool_opts) / sizeof(bool_opts[0]);

    for (size_t i = 0; i < bool_opts_size; i++) {
        if (!strcmp(name, bool_opts[i].name)) {
            return str_to_bool(value, bool_opts[i].out);
        }
    }

    return return_error;
}
