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

#ifndef EbModeDecisionConfigurationProcess_h
#define EbModeDecisionConfigurationProcess_h

#include "EbDefinitions.h"
#include "EbModeDecision.h"
#include "EbSystemResourceManager.h"
#include "EbMdRateEstimation.h"
#include "EbRateControlProcess.h"
#include "EbSequenceControlSet.h"
#include "EbObject.h"
#include "EbInvTransforms.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************************************
     * Defines
     **************************************/
static const uint8_t  depth_offset[4]   = {85, 21, 5, 1};
static const uint32_t ns_blk_offset[10] = {0, 1, 3, 25, 5, 8, 11, 14, 17, 21};
static const uint32_t ns_blk_num[10]    = {1, 2, 2, 4, 3, 3, 3, 3, 4, 4};

typedef struct MdcpLocalBlkStruct {
    uint64_t early_cost;
    Bool   early_split_flag;
    uint32_t split_context;
    Bool   selected_cu;
    Bool   stop_split;
} MdcpLocalBlkStruct;

typedef struct ModeDecisionConfigurationContext {
    EbFifo *rate_control_input_fifo_ptr;
    EbFifo *mode_decision_configuration_output_fifo_ptr;
    uint8_t qp;
    uint8_t qp_index;
} ModeDecisionConfigurationContext;

/**************************************
     * Extern Function Declarations
     **************************************/
EbErrorType mode_decision_configuration_context_ctor(EbThreadContext   *thread_context_ptr,
                                                     const EbEncHandle *enc_handle_ptr,
                                                     int input_index, int output_index);

extern void *mode_decision_configuration_kernel(void *input_ptr);
#ifdef __cplusplus
}
#endif
#endif // EbModeDecisionConfigurationProcess_h
