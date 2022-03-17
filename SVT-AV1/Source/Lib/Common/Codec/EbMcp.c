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

#include <string.h>

#include "EbPictureOperators.h"
#include "common_dsp_rtcd.h"
#include "EbLog.h"

#if (InternalBitDepthIncrement == 0)
#define ChromaOffset4 (1 << (Shift4 - 1))
#else
#define ChromaOffset4 Offset4
#endif
#if (InternalBitDepthIncrement == 0)
#define ChromaMinusOffset1 0
#else
#define ChromaMinusOffset1 MinusOffset1
#endif

/* Pad padding_width pixels on left for a block of height row_height */
void generate_padding_l(EbByte src_pic, uint32_t src_stride, uint32_t row_height,
                        uint32_t padding_width) {
    uint32_t vertical_idx = row_height;
    while (vertical_idx) {
        // left padding
        EB_MEMSET(src_pic - padding_width, *src_pic, padding_width);
        src_pic += src_stride;
        --vertical_idx;
    }
}

/* Pad padding_width pixels on right for a block of height row_height */
void generate_padding_r(EbByte src_pic, uint32_t src_stride, uint32_t row_width,
                        uint32_t row_height, uint32_t padding_width) {
    uint32_t vertical_idx = row_height;
    while (vertical_idx) {
        // right padding
        EB_MEMSET(src_pic + row_width, *(src_pic + row_width - 1), padding_width);
        src_pic += src_stride;
        --vertical_idx;
    }
}

/* Pad padding_height pixels on top for a block of width row_width */
void generate_padding_t(EbByte src_pic, uint32_t src_stride, uint32_t row_width,
                        uint32_t padding_height) {
    uint32_t vertical_idx = padding_height;
    EbByte   temp_src_pic;

    temp_src_pic = src_pic;
    while (vertical_idx) {
        // top part data copy
        temp_src_pic -= src_stride;
        svt_memcpy(temp_src_pic, src_pic, sizeof(uint8_t) * row_width);
        --vertical_idx;
    }
}

/* Pad padding_height pixels in the bottom for a block of width row_width */
void generate_padding_b(EbByte src_pic, uint32_t src_stride, uint32_t row_width,
                        uint32_t row_height, uint32_t padding_height) {
    uint32_t vertical_idx = padding_height;
    EbByte   temp_src_pic, temp_src_pic_1;
    temp_src_pic = temp_src_pic_1 = src_pic + (src_stride * (row_height - 1));
    while (vertical_idx) {
        // bottom part data copy
        temp_src_pic += src_stride;
        svt_memcpy(temp_src_pic, temp_src_pic_1, sizeof(uint8_t) * row_width);
        --vertical_idx;
    }
}

/* left padding for high bit depth */
void generate_padding_l_hbd(EbByte src_pic, uint32_t src_stride, uint32_t row_height,
                            uint32_t padding_width) {
    uint32_t vertical_idx = row_height;
    while (vertical_idx) {
        // left padding
        memset16bit(
            (uint16_t*)(src_pic - padding_width), ((uint16_t*)(src_pic))[0], padding_width >> 1);
        src_pic += src_stride;
        --vertical_idx;
    }
}

/* right padding for high bit depth */
void generate_padding_r_hbd(EbByte src_pic, uint32_t src_stride, uint32_t row_width,
                            uint32_t row_height, uint32_t padding_width) {
    uint32_t vertical_idx = row_height;
    while (vertical_idx) {
        // right padding
        memset16bit((uint16_t*)(src_pic + row_width),
                    ((uint16_t*)(src_pic + row_width - 2))[0],
                    padding_width >> 1);
        src_pic += src_stride;
        --vertical_idx;
    }
}

/** generate_padding()
        is used to pad the target picture. The horizontal padding happens first and then the vertical padding.
 */
void generate_padding(
    EbByte   src_pic, //output paramter, pointer to the source picture to be padded.
    uint32_t src_stride, //input paramter, the stride of the source picture to be padded.
    uint32_t
        original_src_width, //input paramter, the width of the source picture which excludes the padding.
    uint32_t
        original_src_height, //input paramter, the height of the source picture which excludes the padding.
    uint32_t padding_width, //input paramter, the padding width.
    uint32_t padding_height) //input paramter, the padding height.
{
    uint32_t vertical_idx = original_src_height;
    EbByte   temp_src_pic0;
    EbByte   temp_src_pic1;
    EbByte   temp_src_pic2;
    EbByte   temp_src_pic3;

    if (!src_pic) {
        SVT_ERROR("padding NULL pointers\n");
        return;
    }

    temp_src_pic0 = src_pic + padding_width + padding_height * src_stride;
    while (vertical_idx) {
        // horizontal padding
        EB_MEMSET(temp_src_pic0 - padding_width, *temp_src_pic0, padding_width);
        EB_MEMSET(temp_src_pic0 + original_src_width,
                  *(temp_src_pic0 + original_src_width - 1),
                  padding_width);

        temp_src_pic0 += src_stride;
        --vertical_idx;
    }

    // vertical padding
    vertical_idx  = padding_height;
    temp_src_pic0 = src_pic + padding_height * src_stride;
    temp_src_pic1 = src_pic + (padding_height + original_src_height - 1) * src_stride;
    temp_src_pic2 = temp_src_pic0;
    temp_src_pic3 = temp_src_pic1;
    while (vertical_idx) {
        // top part data copy
        temp_src_pic2 -= src_stride;
        svt_memcpy(
            temp_src_pic2, temp_src_pic0, sizeof(uint8_t) * src_stride); // uint8_t to be modified
        // bottom part data copy
        temp_src_pic3 += src_stride;
        svt_memcpy(
            temp_src_pic3, temp_src_pic1, sizeof(uint8_t) * src_stride); // uint8_t to be modified
        --vertical_idx;
    }

    return;
}
void generate_padding_compressed_10bit(
    EbByte   src_pic, //output paramter, pointer to the source picture to be padded.
    uint32_t src_stride, //input paramter, the stride of the source picture to be padded.
    uint32_t
        original_src_width, //input paramter, the width of the source picture which excludes the padding.
    uint32_t
        original_src_height, //input paramter, the height of the source picture which excludes the padding.
    uint32_t padding_width, //input paramter, the padding width.
    uint32_t padding_height) //input paramter, the padding height.
{
    uint32_t vertical_idx = original_src_height;
    EbByte   temp_src_pic0;
    EbByte   temp_src_pic1;
    EbByte   temp_src_pic2;
    EbByte   temp_src_pic3;

    if (!src_pic) {
        SVT_ERROR("padding NULL pointers\n");
        return;
    }
    temp_src_pic0 = src_pic + padding_width / 4 + padding_height * src_stride;

    for (uint32_t row = 0; row < original_src_height; row++) {
        uint8_t left_pixel, right_pixel, new_left_byte, new_right_byte;
        left_pixel  = (temp_src_pic0[0] >> 6) & 0x03;
        right_pixel = temp_src_pic0[original_src_width / 4 - 1] & 0x03;

        new_left_byte = ((left_pixel << 6) & 0xC0) | ((left_pixel << 4) & 0x30) |
            ((left_pixel << 2) & 0x0C) | left_pixel;
        new_right_byte = ((right_pixel << 6) & 0xC0) | ((right_pixel << 4) & 0x30) |
            ((right_pixel << 2) & 0x0C) | right_pixel;

        EB_MEMSET(temp_src_pic0 - padding_width / 4, new_left_byte, padding_width / 4);

        EB_MEMSET(temp_src_pic0 + original_src_width / 4, new_right_byte, padding_width / 4);

        temp_src_pic0 += src_stride;
    }

    // vertical padding
    vertical_idx  = padding_height;
    temp_src_pic0 = src_pic + padding_height * src_stride;
    temp_src_pic1 = src_pic + (padding_height + original_src_height - 1) * src_stride;
    temp_src_pic2 = temp_src_pic0;
    temp_src_pic3 = temp_src_pic1;
    while (vertical_idx) {
        // top part data copy
        temp_src_pic2 -= src_stride;
        svt_memcpy(
            temp_src_pic2, temp_src_pic0, sizeof(uint8_t) * src_stride); // uint8_t to be modified
        // bottom part data copy
        temp_src_pic3 += src_stride;
        svt_memcpy(
            temp_src_pic3, temp_src_pic1, sizeof(uint8_t) * src_stride); // uint8_t to be modified
        --vertical_idx;
    }

    return;
}
/** generate_padding16_bit()
is used to pad the target picture. The horizontal padding happens first and then the vertical padding.
*/
// TODO: generate_padding() and generate_padding16() functions are not aligned, inputs according to comments are wrong
void generate_padding16_bit(
    uint16_t* src_pic, //output paramter, pointer to the source picture to be padded.
    uint32_t  src_stride, //input paramter, the stride of the source picture to be padded.
    uint32_t
        original_src_width, //input paramter, the width of the source picture which excludes the padding.
    uint32_t
        original_src_height, //input paramter, the height of the source picture which excludes the padding.
    uint32_t padding_width, //input paramter, the padding width.
    uint32_t padding_height) //input paramter, the padding height.
{
    uint32_t  vertical_idx = original_src_height;
    uint16_t* temp_src_pic0;
    uint16_t* temp_src_pic1;
    uint16_t* temp_src_pic2;
    uint16_t* temp_src_pic3;

    temp_src_pic0 = src_pic + padding_width + padding_height * src_stride;
    while (vertical_idx) {
        // horizontal padding
        //EB_MEMSET(temp_src_pic0 - padding_width, temp_src_pic0, padding_width);
        memset16bit(temp_src_pic0 - padding_width, temp_src_pic0[0], padding_width);
        memset16bit(temp_src_pic0 + original_src_width,
                    (temp_src_pic0 + original_src_width - 1)[0],
                    padding_width);

        temp_src_pic0 += src_stride;
        --vertical_idx;
    }

    // vertical padding
    vertical_idx  = padding_height;
    temp_src_pic0 = src_pic + padding_height * src_stride;
    temp_src_pic1 = src_pic + (padding_height + original_src_height - 1) * src_stride;
    temp_src_pic2 = temp_src_pic0;
    temp_src_pic3 = temp_src_pic1;
    while (vertical_idx) {
        // top part data copy
        temp_src_pic2 -= src_stride;
        svt_memcpy(temp_src_pic2, temp_src_pic0, sizeof(uint16_t) * src_stride);
        // bottom part data copy
        temp_src_pic3 += src_stride;
        svt_memcpy(temp_src_pic3, temp_src_pic1, sizeof(uint16_t) * src_stride);
        --vertical_idx;
    }

    return;
}

/** pad_input_picture()
is used to pad the input picture in order to get . The horizontal padding happens first and then the vertical padding.
*/
void pad_input_picture(
    EbByte   src_pic, //output paramter, pointer to the source picture to be padded.
    uint32_t src_stride, //input paramter, the stride of the source picture to be padded.
    uint32_t
        original_src_width, //input paramter, the width of the source picture which excludes the padding.
    uint32_t
        original_src_height, //input paramter, the height of the source picture which excludes the padding.
    uint32_t pad_right, //input paramter, the padding right.
    uint32_t pad_bottom) //input paramter, the padding bottom.
{
    uint32_t vertical_idx;
    EbByte   temp_src_pic0;

    if (!src_pic) {
        SVT_ERROR("padding NULL pointers\n");
        return;
    }

    if (pad_right) {
        // Add padding @ the right
        vertical_idx  = original_src_height;
        temp_src_pic0 = src_pic;

        while (vertical_idx) {
            EB_MEMSET(temp_src_pic0 + original_src_width,
                      *(temp_src_pic0 + original_src_width - 1),
                      pad_right);
            temp_src_pic0 += src_stride;
            --vertical_idx;
        }
    }

    if (pad_bottom) {
        EbByte temp_src_pic1;
        // Add padding @ the bottom
        vertical_idx  = pad_bottom;
        temp_src_pic0 = src_pic + (original_src_height - 1) * src_stride;
        temp_src_pic1 = temp_src_pic0;

        while (vertical_idx) {
            temp_src_pic1 += src_stride;
            svt_memcpy(
                temp_src_pic1, temp_src_pic0, sizeof(uint8_t) * (original_src_width + pad_right));
            --vertical_idx;
        }
    }

    return;
}

/** pad_input_picture_16bit()
is used to pad the input picture in order to get . The horizontal padding happens first and then the vertical padding.
*/
void pad_input_picture_16bit(
    uint16_t* src_pic, //output paramter, pointer to the source picture to be padded.
    uint32_t  src_stride, //input paramter, the stride of the source picture to be padded.
    uint32_t
        original_src_width, //input paramter, the width of the source picture which excludes the padding.
    uint32_t
        original_src_height, //input paramter, the height of the source picture which excludes the padding.
    uint32_t pad_right, //input paramter, the padding right.
    uint32_t pad_bottom) //input paramter, the padding bottom.
{
    uint32_t  vertical_idx;
    uint16_t* temp_src_pic0;

    if (pad_right) {
        // Add padding @ the right
        vertical_idx  = original_src_height;
        temp_src_pic0 = src_pic;

        while (vertical_idx) {
            memset16bit(temp_src_pic0 + original_src_width,
                        *(temp_src_pic0 + original_src_width - 1),
                        pad_right);
            temp_src_pic0 += src_stride;
            --vertical_idx;
        }
    }

    if (pad_bottom) {
        uint16_t* temp_src_pic1;
        // Add padding @ the bottom
        vertical_idx  = pad_bottom;
        temp_src_pic0 = (uint16_t*)(src_pic + (original_src_height - 1) * src_stride);
        temp_src_pic1 = temp_src_pic0;

        while (vertical_idx) {
            temp_src_pic1 += src_stride;
            svt_memcpy(
                temp_src_pic1, temp_src_pic0, sizeof(uint16_t) * (original_src_width + pad_right));
            --vertical_idx;
        }
    }

    return;
}
