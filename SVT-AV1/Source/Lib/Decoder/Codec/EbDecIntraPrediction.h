/*
* Copyright(c) 2019 Netflix, Inc.
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef EbDecIntraPrediction_h
#define EbDecIntraPrediction_h

#ifdef __cplusplus
extern "C" {
#endif

// Do we need to save the luma pixels from the current block,
// for a possible future CfL prediction?
CflAllowedType store_cfl_required(const EbColorConfig *cc, PartitionInfo *xd,
                                  int32_t is_chroma_ref);

void svt_av1_predict_intra(DecModCtxt *dec_mod_ctxt, PartitionInfo *part_info, int32_t plane,
                           TxSize tx_size, TileInfo *td, void *pv_blk_recon_buf,
                           int32_t recon_stride, EbBitDepthEnum bit_depth, int32_t blk_mi_col_off,
                           int32_t blk_mi_row_off);

void svtav1_predict_intra_block(PartitionInfo *xd, int32_t plane, TxSize tx_size, TileInfo *td,
                                void *pv_pred_buf, int32_t pred_stride, void *top_neigh_array,
                                void *left_neigh_array, int32_t ref_stride, SeqHeader *seq_header,
                                const PredictionMode mode, int32_t blk_mi_col_off,
                                int32_t blk_mi_row_off, EbBitDepthEnum bit_depth, Bool is_16bit);

void svt_cfl_store_tx(PartitionInfo *xd, CflCtx *cfl_ctx, int row, int col, TxSize tx_size,
                      BlockSize bsize, EbColorConfig *cc, uint8_t *dst_buff, uint32_t dst_stride,
                      Bool is_16bit);

#ifdef __cplusplus
}
#endif
#endif // EbDecIntraPrediction_h
