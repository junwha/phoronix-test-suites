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

// Command line argument parsing

#ifndef EbDecParamParser_h
#define EbDecParamParser_h

/***************************************
 * Includes
 ***************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "EbSvtAv1Dec.h"
#include "EbFileUtils.h"

#ifdef _WIN32
#define FOPEN(f, s, m) fopen_s(&f, s, m)
#else
#define FOPEN(f, s, m) f = fopen(s, m)
#endif

/**********************************
  * CLI options
  **********************************/
#define HELP_TOKEN "-help"
#define COMMAND_LINE_MAX_SIZE 2048
#define INPUT_FILE_TOKEN "-i"
#define OUTPUT_FILE_TOKEN "-o"
#define SKIP_FRAME_TOKEN "-skip"
#define LIMIT_FRAME_TOKEN "-limit"
#define BIT_DEPTH_TOKEN "-bit-depth"
#define DECODER_16BIT_PIPELINE "-16bit-pipeline"
#define PIC_WIDTH_TOKEN "-w"
#define PIC_HEIGHT_TOKEN "-h"
#define COLOUR_SPACE_TOKEN "-colour-space"
#define THREADS_TOKEN "-threads"
#define FRAME_PLL_TOKEN "-parallel-frames"
#define MD5_SUPPORT_TOKEN "-md5"
#define FPS_FRM_TOKEN "-fps-frm"
#define FPS_SUMMARY_TOKEN "-fps-summary"
#define FILM_GRAIN_TOKEN "-skip-film-grain"
#define ANNEX_B_TOKEN "-annex-b"
#define MAX_NUM_TOKENS 200

/**********************************
 * Config Entry Struct
 **********************************/
typedef struct ConfigEntry {
    const char *token;
    const char *name;
    Bool      value_required;
    void (*scf)(const char *, EbSvtAv1DecConfiguration *);
} ConfigEntry;

EbErrorType read_command_line(int32_t argc, char *const argv[], EbSvtAv1DecConfiguration *configs,
                              CliInput *cli, ObuDecInputContext *obu_ctx);

#endif
