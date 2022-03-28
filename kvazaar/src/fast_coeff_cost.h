/*****************************************************************************
 * This file is part of Kvazaar HEVC encoder.
 *
 * Copyright (c) 2021, Tampere University, ITU/ISO/IEC, project contributors
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * * Redistributions in binary form must reproduce the above copyright notice, this
 *   list of conditions and the following disclaimer in the documentation and/or
 *   other materials provided with the distribution.
 * 
 * * Neither the name of the Tampere University or ITU/ISO/IEC nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * INCLUDING NEGLIGENCE OR OTHERWISE ARISING IN ANY WAY OUT OF THE USE OF THIS
 ****************************************************************************/

#ifndef FAST_COEFF_COST_H_
#define FAST_COEFF_COST_H_

#include <stdio.h>
#include "kvazaar.h"
// #include "encoderstate.h"

#define MAX_FAST_COEFF_COST_QP 50

typedef struct {
  uint64_t wts_by_qp[MAX_FAST_COEFF_COST_QP];
} fast_coeff_table_t;

// Weights for 4 buckets (coeff 0, coeff 1, coeff 2, coeff >= 3), for QPs from
// 0 to MAX_FAST_COEFF_COST_QP
static const double default_fast_coeff_cost_wts[][4] = {
  // Just extend it by stretching the first actual values..
  {0.164240, 4.161530, 3.509033, 6.928047},
  {0.164240, 4.161530, 3.509033, 6.928047},
  {0.164240, 4.161530, 3.509033, 6.928047},
  {0.164240, 4.161530, 3.509033, 6.928047},
  {0.164240, 4.161530, 3.509033, 6.928047},
  {0.164240, 4.161530, 3.509033, 6.928047},
  {0.164240, 4.161530, 3.509033, 6.928047},
  {0.164240, 4.161530, 3.509033, 6.928047},
  {0.164240, 4.161530, 3.509033, 6.928047},
  {0.164240, 4.161530, 3.509033, 6.928047},
  // up to here
  {0.164240, 4.161530, 3.509033, 6.928047},
  {0.162844, 4.055940, 3.564467, 6.861493},
  {0.128729, 4.311973, 3.942837, 6.935403},
  {0.110956, 4.433190, 3.945753, 6.877697},
  {0.095026, 4.483547, 4.194173, 6.781540},
  {0.075046, 4.633703, 4.084193, 6.698600},
  {0.052426, 4.967223, 4.027210, 6.549197},
  {0.040219, 5.141820, 3.982650, 6.461557},
  {0.035090, 5.192493, 3.830950, 6.418477},
  {0.029845, 5.211647, 3.815457, 6.345440},
  {0.023522, 5.322213, 3.816537, 6.360677},
  {0.021305, 5.225923, 3.842700, 6.325787},
  {0.015878, 5.183090, 3.956003, 6.329680},
  {0.010430, 5.099230, 4.176803, 6.305400},
  {0.008433, 5.030257, 4.237587, 6.270133},
  {0.006500, 4.969247, 4.339397, 6.217827},
  {0.004929, 4.923500, 4.442413, 6.183523},
  {0.003715, 4.915583, 4.429090, 6.125320},
  {0.003089, 4.883907, 4.562790, 6.156447},
  {0.002466, 4.881063, 4.629883, 6.142643},
  {0.002169, 4.882493, 4.646313, 6.127663},
  {0.002546, 4.793337, 4.837413, 6.199270},
  {0.001314, 4.808853, 4.828337, 6.243437},
  {0.001154, 4.862603, 4.846883, 6.205523},
  {0.000984, 4.866403, 4.859330, 6.240893},
  {0.000813, 4.856633, 4.924527, 6.293413},
  {0.001112, 4.789260, 5.009880, 6.433540},
  {0.000552, 4.760747, 5.090447, 6.599380},
  {0.000391, 4.961447, 5.111033, 6.756370},
  {0.000332, 4.980953, 5.138127, 6.867420},
  {0.000201, 5.181957, 4.740160, 6.460997},
  {0.000240, 5.185390, 4.874840, 6.819093},
  {0.000130, 5.270350, 4.734213, 6.826240},
  {0.000104, 5.371937, 4.595087, 6.659253},
  {0.000083, 5.362000, 4.617470, 6.837770},
  {0.000069, 5.285997, 4.754993, 7.159043},
  {0.000049, 5.488470, 4.396107, 6.727357},
  {0.000058, 4.958940, 4.580460, 6.477740},
  {0.000028, 5.521253, 4.440493, 7.205017},
  {0.000000, 0.000000, 0.000000, 0.000000},
  {0.000019, 5.811260, 4.399110, 7.336310},
};

typedef struct encoder_state_t encoder_state_t;

int kvz_fast_coeff_table_parse(fast_coeff_table_t *fast_coeff_table, FILE *fast_coeff_table_f);
void kvz_fast_coeff_use_default_table(fast_coeff_table_t *fast_coeff_table);
uint64_t kvz_fast_coeff_get_weights(const encoder_state_t *state);

#endif // FAST_COEFF_COST_H_
