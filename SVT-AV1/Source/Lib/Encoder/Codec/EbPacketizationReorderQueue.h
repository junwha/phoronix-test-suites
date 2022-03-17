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

#ifndef EbPacketizationReorderQueue_h
#define EbPacketizationReorderQueue_h

#include "EbDefinitions.h"
#include "EbSystemResourceManager.h"
#include "EbPredictionStructure.h"
#include "EbObject.h"
#include "EbEntropyCodingObject.h"
#ifdef __cplusplus
extern "C" {
#endif
/************************************************
     * Packetization Reorder Queue Entry
     ************************************************/
typedef struct PacketizationReorderEntry {
    EbDctor          dctor;
    uint64_t         picture_number;
    EbObjectWrapper *output_stream_wrapper_ptr;

    EbLinkedListNode *out_meta_data;

    uint64_t start_time_seconds;
    uint64_t start_time_u_seconds;

    uint8_t    slice_type;
    uint64_t   ref_poc_list0;
    uint64_t   ref_poc_list1;
    uint64_t   ref_poc_array[7];
    uint64_t   poc;
    uint64_t   total_num_bits;
    FrameType  frame_type;
    Av1RpsNode av1_ref_signal;
    Bool     show_frame;
    Bool     has_show_existing;
    uint8_t    show_existing_frame;
    //small size bitstream for show existing frame
    Bitstream *bitstream_ptr;
    //valid when has_show_existing is true
    int64_t                  next_pts;
    uint8_t                  is_alt_ref;
    struct SvtMetadataArray *metadata;
} PacketizationReorderEntry;

extern EbErrorType packetization_reorder_entry_ctor(PacketizationReorderEntry *entry_ptr,
                                                    uint32_t                   picture_number);

#ifdef __cplusplus
}
#endif
#endif //EbPacketizationReorderQueue_h
