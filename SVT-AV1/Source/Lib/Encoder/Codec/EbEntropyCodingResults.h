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

#ifndef EbEntropyCodingResults_h
#define EbEntropyCodingResults_h

#include "EbDefinitions.h"
#include "EbSystemResourceManager.h"
#include "EbObject.h"
#ifdef __cplusplus
extern "C" {
#endif
/**************************************
     * Process Results
     **************************************/
typedef struct EntropyCodingResults {
    EbDctor          dctor;
    EbObjectWrapper *pcs_wrapper_ptr;
} EntropyCodingResults;

typedef struct EntropyCodingResultsInitData {
    uint32_t junk;
} EntropyCodingResultsInitData;

/**************************************
     * Extern Function Declarations
     **************************************/
extern EbErrorType entropy_coding_results_creator(EbPtr *object_dbl_ptr,
                                                  EbPtr  object_init_data_ptr);

#ifdef __cplusplus
}
#endif
#endif // EbEntropyCodingResults_h
