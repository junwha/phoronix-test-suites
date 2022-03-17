/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */
// workaround to eliminate the compiling warning on linux
// The macro will conflict with definition in gtest.h
#ifdef __USE_GNU
#undef __USE_GNU  // defined in EbThreads.h
#endif
#ifdef _GNU_SOURCE
#undef _GNU_SOURCE  // defined in EbThreads.h
#endif
#include "EbDefinitions.h"
#include "EbUnitTestUtility.h"
#include "EbWarpedMotion.h"
#include "warp_filter_test_util.h"
#include "convolve.h"

using std::make_tuple;
using std::tuple;

namespace libaom_test {

static const int quant_dist_lookup_table[2][4][2] = {
    {{9, 7}, {11, 5}, {12, 4}, {13, 3}},
    {{7, 9}, {5, 11}, {4, 12}, {3, 13}},
};

int32_t random_warped_param(svt_av1_test_tool::SVTRandom *rnd, int bits) {
    // 1 in 8 chance of generating zero (arbitrarily chosen)
    if (((rnd->Rand8()) & 7) == 0)
        return 0;
    // Otherwise, enerate uniform values in the range
    // [-(1 << bits), 1] U [1, 1<<bits]
    int32_t v = 1 + (rnd->Rand16() & ((1 << bits) - 1));
    if ((rnd->Rand8()) & 1)
        return -v;
    return v;
}

void generate_warped_model(svt_av1_test_tool::SVTRandom *rnd, int32_t *mat,
                           int16_t *alpha, int16_t *beta, int16_t *gamma,
                           int16_t *delta, const int is_alpha_zero,
                           const int is_beta_zero, const int is_gamma_zero,
                           const int is_delta_zero) {
    while (1) {
        int rnd8 = rnd->Rand8() & 3;
        mat[0] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS + 6);
        mat[1] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS + 6);
        mat[2] = (random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3)) +
                 (1 << WARPEDMODEL_PREC_BITS);
        mat[3] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3);

        if (rnd8 <= 1) {
            // AFFINE
            mat[4] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3);
            mat[5] = (random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3)) +
                     (1 << WARPEDMODEL_PREC_BITS);
        } else if (rnd8 == 2) {
            mat[4] = -mat[3];
            mat[5] = mat[2];
        } else {
            mat[4] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3);
            mat[5] = (random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3)) +
                     (1 << WARPEDMODEL_PREC_BITS);
            if (is_alpha_zero == 1)
                mat[2] = 1 << WARPEDMODEL_PREC_BITS;
            if (is_beta_zero == 1)
                mat[3] = 0;
            if (is_gamma_zero == 1)
                mat[4] = 0;
            if (is_delta_zero == 1)
                mat[5] = (((int64_t)mat[3] * mat[4] + (mat[2] / 2)) / mat[2]) +
                         (1 << WARPEDMODEL_PREC_BITS);
        }

        // Calculate the derived parameters and check that they are suitable
        // for the warp filter.
        assert(mat[2] != 0);

        *alpha = static_cast<int16_t>(
            clamp(mat[2] - (1 << WARPEDMODEL_PREC_BITS), INT16_MIN, INT16_MAX));
        *beta = static_cast<int16_t>(clamp(mat[3], INT16_MIN, INT16_MAX));
        *gamma = static_cast<int16_t>(
            clamp(((int64_t)mat[4] * (1 << WARPEDMODEL_PREC_BITS)) / mat[2],
                  INT16_MIN,
                  INT16_MAX));
        *delta = static_cast<int16_t>(clamp(
            mat[5] - (((int64_t)mat[3] * mat[4] + (mat[2] / 2)) / mat[2]) -
                (1 << WARPEDMODEL_PREC_BITS),
            INT16_MIN,
            INT16_MAX));

        if ((4 * abs(*alpha) + 7 * abs(*beta) >=
             (1 << WARPEDMODEL_PREC_BITS)) ||
            (4 * abs(*gamma) + 4 * abs(*delta) >= (1 << WARPEDMODEL_PREC_BITS)))
            continue;

        *alpha = ROUND_POWER_OF_TWO_SIGNED(*alpha, WARP_PARAM_REDUCE_BITS) *
                 (1 << WARP_PARAM_REDUCE_BITS);
        *beta = ROUND_POWER_OF_TWO_SIGNED(*beta, WARP_PARAM_REDUCE_BITS) *
                (1 << WARP_PARAM_REDUCE_BITS);
        *gamma = ROUND_POWER_OF_TWO_SIGNED(*gamma, WARP_PARAM_REDUCE_BITS) *
                 (1 << WARP_PARAM_REDUCE_BITS);
        *delta = ROUND_POWER_OF_TWO_SIGNED(*delta, WARP_PARAM_REDUCE_BITS) *
                 (1 << WARP_PARAM_REDUCE_BITS);

        // We have a valid model, so finish
        return;
    }
}

namespace AV1WarpFilter {
::testing::internal::ParamGenerator<WarpTestParams> BuildParams(
    warp_affine_func filter) {
    WarpTestParam params[] = {
        make_tuple(4, 4, 50000, filter),
        make_tuple(8, 8, 50000, filter),
        make_tuple(64, 64, 1000, filter),
        make_tuple(4, 16, 20000, filter),
        make_tuple(32, 8, 10000, filter),
    };
    return ::testing::Combine(::testing::ValuesIn(params),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1));
}

AV1WarpFilterTest::~AV1WarpFilterTest() {
}
void AV1WarpFilterTest::SetUp() {
    rnd_ = new svt_av1_test_tool::SVTRandom(0, (1 << 8) - 1);
}

void AV1WarpFilterTest::TearDown() {
    delete rnd_;
    aom_clear_system_state();
}

void AV1WarpFilterTest::RunSpeedTest(warp_affine_func test_impl) {
    const int w = 128, h = 128;
    const int border = 16;
    const int stride = w + 2 * border;
    WarpTestParam params = TEST_GET_PARAM(0);
    const int out_w = std::get<0>(params), out_h = std::get<1>(params);
    const int is_alpha_zero = TEST_GET_PARAM(1);
    const int is_beta_zero = TEST_GET_PARAM(2);
    const int is_gamma_zero = TEST_GET_PARAM(3);
    const int is_delta_zero = TEST_GET_PARAM(4);
    int sub_x, sub_y;
    const int bd = 8;
    const int32_t ref = 0;

    uint8_t *input_ = new uint8_t[h * stride];
    uint8_t *input = input_ + border;

    // The warp functions always write rows with widths that are multiples of 8.
    // So to avoid a buffer overflow, we may need to pad rows to a multiple
    // of 8.
    int output_n = ((out_w + 7) & ~7) * out_h;
    uint8_t *output = new uint8_t[output_n];
    int32_t mat[8];
    int16_t alpha, beta, gamma, delta;
    ConvBufType *dsta = new ConvBufType[output_n];
    generate_warped_model(rnd_,
                          mat,
                          &alpha,
                          &beta,
                          &gamma,
                          &delta,
                          is_alpha_zero,
                          is_beta_zero,
                          is_gamma_zero,
                          is_delta_zero);

    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c)
            input[r * stride + c] = rnd_->Rand8();
    for (int r = 0; r < h; ++r) {
        memset(input + r * stride - border, input[r * stride], border);
        memset(input + r * stride + w, input[r * stride + (w - 1)], border);
    }

    sub_x = 0;
    sub_y = 0;
    int do_average = 0;

    ConvolveParams conv_params =
        get_conv_params_no_round(ref, do_average, 0, dsta, out_w, 1, bd);
    conv_params.use_jnt_comp_avg = 0;

    const int num_loops = 1000000000 / (out_w + out_h);
    double elapsed_time;
    uint64_t start_time_seconds, start_time_useconds;
    uint64_t finish_time_seconds, finish_time_useconds;

    svt_av1_get_time(&start_time_seconds, &start_time_useconds);

    for (int i = 0; i < num_loops; ++i)
        test_impl(mat,
                  input,
                  w,
                  h,
                  stride,
                  output,
                  32,
                  32,
                  out_w,
                  out_h,
                  out_w,
                  sub_x,
                  sub_y,
                  &conv_params,
                  alpha,
                  beta,
                  gamma,
                  delta);

    svt_av1_get_time(&finish_time_seconds, &finish_time_useconds);
    elapsed_time =
        svt_av1_compute_overall_elapsed_time_ms(start_time_seconds,
                                                start_time_useconds,
                                                finish_time_seconds,
                                                finish_time_useconds);
    printf("warp %3dx%-3d: %7.2f ns\n",
           out_w,
           out_h,
           1000.0 * elapsed_time / num_loops);

    delete[] input_;
    delete[] output;
    delete[] dsta;
}

void AV1WarpFilterTest::RunCheckOutput(warp_affine_func test_impl) {
    const int w = 128, h = 128;
    const int border = 16;
    const int stride = w + 2 * border;
    WarpTestParam params = TEST_GET_PARAM(0);
    const int is_alpha_zero = TEST_GET_PARAM(1);
    const int is_beta_zero = TEST_GET_PARAM(2);
    const int is_gamma_zero = TEST_GET_PARAM(3);
    const int is_delta_zero = TEST_GET_PARAM(4);
    const int out_w = std::get<0>(params), out_h = std::get<1>(params);
    const int num_iters = std::get<2>(params);
    int i, j, sub_x, sub_y;
    const int bd = 8;
    const int32_t ref = 0;

    // The warp functions always write rows with widths that are multiples of 8.
    // So to avoid a buffer overflow, we may need to pad rows to a multiple
    // of 8.
    int output_n = ((out_w + 7) & ~7) * out_h;
    uint8_t *input_ = new uint8_t[h * stride];
    uint8_t *input = input_ + border;
    uint8_t *output = new uint8_t[output_n];
    uint8_t *output2 = new uint8_t[output_n];
    int32_t mat[8];
    int16_t alpha, beta, gamma, delta;
    ConvolveParams conv_params = get_conv_params(ref, 0, 0, bd);
    ConvBufType *dsta = new ConvBufType[output_n];
    ConvBufType *dstb = new ConvBufType[output_n];
    for (i = 0; i < output_n; ++i) {
        output[i] = output2[i] = rnd_->Rand8();
        dsta[i] = dstb[i] = 0;
    }

    for (i = 0; i < num_iters; ++i) {
        // Generate an input block and extend its borders horizontally
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
                input[r * stride + c] = rnd_->Rand8();
        for (int r = 0; r < h; ++r) {
            memset(input + r * stride - border, input[r * stride], border);
            memset(input + r * stride + w, input[r * stride + (w - 1)], border);
        }
        const int use_no_round = rnd_->Rand8() & 1;
        for (sub_x = 0; sub_x < 2; ++sub_x)
            for (sub_y = 0; sub_y < 2; ++sub_y) {
                generate_warped_model(rnd_,
                                      mat,
                                      &alpha,
                                      &beta,
                                      &gamma,
                                      &delta,
                                      is_alpha_zero,
                                      is_beta_zero,
                                      is_gamma_zero,
                                      is_delta_zero);

                for (int ii = 0; ii < 2; ++ii) {
                    for (int jj = 0; jj < 5; ++jj) {
                        for (int do_average = 0; do_average <= 1;
                             ++do_average) {
                            if (use_no_round) {
                                conv_params = get_conv_params_no_round(
                                    ref, do_average, 0, dsta, out_w, 1, bd);
                            } else {
                                conv_params = get_conv_params(ref, 0, 0, bd);
                            }
                            if (jj >= 4) {
                                conv_params.use_jnt_comp_avg = 0;
                            } else {
                                conv_params.use_jnt_comp_avg = 1;
                                conv_params.fwd_offset =
                                    quant_dist_lookup_table[ii][jj][0];
                                conv_params.bck_offset =
                                    quant_dist_lookup_table[ii][jj][1];
                            }
                            svt_av1_warp_affine_c(mat,
                                                  input,
                                                  w,
                                                  h,
                                                  stride,
                                                  output,
                                                  32,
                                                  32,
                                                  out_w,
                                                  out_h,
                                                  out_w,
                                                  sub_x,
                                                  sub_y,
                                                  &conv_params,
                                                  alpha,
                                                  beta,
                                                  gamma,
                                                  delta);
                            if (use_no_round) {
                                conv_params = get_conv_params_no_round(
                                    ref, do_average, 0, dstb, out_w, 1, bd);
                            }
                            if (jj >= 4) {
                                conv_params.use_jnt_comp_avg = 0;
                            } else {
                                conv_params.use_jnt_comp_avg = 1;
                                conv_params.fwd_offset =
                                    quant_dist_lookup_table[ii][jj][0];
                                conv_params.bck_offset =
                                    quant_dist_lookup_table[ii][jj][1];
                            }
                            test_impl(mat,
                                      input,
                                      w,
                                      h,
                                      stride,
                                      output2,
                                      32,
                                      32,
                                      out_w,
                                      out_h,
                                      out_w,
                                      sub_x,
                                      sub_y,
                                      &conv_params,
                                      alpha,
                                      beta,
                                      gamma,
                                      delta);
                            if (use_no_round) {
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(dsta[j], dstb[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(output[j], output2[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                            } else {
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(output[j], output2[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                            }
                        }
                    }
                }
            }
    }
    delete[] input_;
    delete[] output;
    delete[] output2;
    delete[] dsta;
    delete[] dstb;
}
}  // namespace AV1WarpFilter

namespace AV1HighbdWarpFilter {
::testing::internal::ParamGenerator<HighbdWarpTestParams> BuildParams(
    highbd_warp_affine_func filter) {
    const HighbdWarpTestParam params[] = {
        make_tuple(4, 4, 100, 8, filter),
        make_tuple(8, 8, 100, 8, filter),
        make_tuple(64, 64, 100, 8, filter),
        make_tuple(4, 16, 100, 8, filter),
        make_tuple(32, 8, 100, 8, filter),
        make_tuple(4, 4, 100, 10, filter),
        make_tuple(8, 8, 100, 10, filter),
        make_tuple(64, 64, 100, 10, filter),
        make_tuple(4, 16, 100, 10, filter),
        make_tuple(32, 8, 100, 10, filter),
        make_tuple(4, 4, 100, 12, filter),
        make_tuple(8, 8, 100, 12, filter),
        make_tuple(64, 64, 100, 12, filter),
        make_tuple(4, 16, 100, 12, filter),
        make_tuple(32, 8, 100, 12, filter),
    };
    return ::testing::Combine(::testing::ValuesIn(params),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1));
}

AV1HighbdWarpFilterTest::~AV1HighbdWarpFilterTest() {
}
void AV1HighbdWarpFilterTest::SetUp() {
    rnd_ = new svt_av1_test_tool::SVTRandom(0, (1 << 12) - 1);
}

void AV1HighbdWarpFilterTest::TearDown() {
    delete rnd_;
    aom_clear_system_state();
}

void AV1HighbdWarpFilterTest::RunSpeedTest(highbd_warp_affine_func test_impl) {
    const int w = 128, h = 128;
    const int border = 16;
    const int stride8b = w + 2 * border;
    const int stride2b = w + 2 * border;
    HighbdWarpTestParam param = TEST_GET_PARAM(0);
    const int is_alpha_zero = TEST_GET_PARAM(1);
    const int is_beta_zero = TEST_GET_PARAM(2);
    const int is_gamma_zero = TEST_GET_PARAM(3);
    const int is_delta_zero = TEST_GET_PARAM(4);
    const int out_w = std::get<0>(param), out_h = std::get<1>(param);
    const int bd = std::get<3>(param);
    const int32_t ref = 0;
    const int mask = (1 << bd) - 1;
    int sub_x, sub_y;

    // The warp functions always write rows with widths that are multiples of 8.
    // So to avoid a buffer overflow, we may need to pad rows to a multiple
    // of 8.
    int output_n = ((out_w + 7) & ~7) * out_h;
    uint8_t *input8b_ = new uint8_t[h * stride8b];
    uint8_t *input8b = input8b_ + border;
    uint8_t *input2b_ = new uint8_t[h * stride2b];
    uint8_t *input2b = input2b_ + border;
    uint16_t *output = new uint16_t[output_n];
    int32_t mat[8];
    int16_t alpha, beta, gamma, delta;
    ConvolveParams conv_params = get_conv_params(ref, 0, 0, bd);
    ConvBufType *dsta = new ConvBufType[output_n];

    generate_warped_model(rnd_,
                          mat,
                          &alpha,
                          &beta,
                          &gamma,
                          &delta,
                          is_alpha_zero,
                          is_beta_zero,
                          is_gamma_zero,
                          is_delta_zero);
    // Generate an input block and extend its borders horizontally
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            uint16_t val = rnd_->Rand16() & mask;
            input8b[r * stride8b + c] = val >> 2;
            input2b[r * stride2b + c] = (val & 3) << 6;
        }
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < border; ++c) {
            input8b[r * stride8b - border + c] = input8b[r * stride8b];
            input8b[r * stride8b + w + c] = input8b[r * stride8b + (w - 1)];
            input2b[r * stride2b - border + c] = input2b[r * stride2b];
            input2b[r * stride2b + w + c] = input2b[r * stride2b + (w - 1)];
        }
    }

    sub_x = 0;
    sub_y = 0;
    int do_average = 0;
    conv_params.use_jnt_comp_avg = 0;
    conv_params =
        get_conv_params_no_round(ref, do_average, 0, dsta, out_w, 1, bd);

    const int num_loops = 50000000 / (out_w * out_h);
    double elapsed_time_tst;
    double elapsed_time_ref;
    uint64_t start_time_seconds_tst, start_time_useconds_tst;
    uint64_t finish_time_seconds_tst, finish_time_useconds_tst;
    uint64_t start_time_seconds_ref, start_time_useconds_ref;
    uint64_t finish_time_seconds_ref, finish_time_useconds_ref;

    svt_av1_get_time(&start_time_seconds_tst, &start_time_useconds_tst);

    for (int i = 0; i < num_loops; ++i)
        test_impl(mat,
                  input8b,
                  input2b,
                  w,
                  h,
                  stride8b,
                  stride2b,
                  output,
                  32,
                  32,
                  out_w,
                  out_h,
                  out_w,
                  sub_x,
                  sub_y,
                  bd,
                  &conv_params,
                  alpha,
                  beta,
                  gamma,
                  delta);

    svt_av1_get_time(&finish_time_seconds_tst, &finish_time_useconds_tst);
    elapsed_time_tst =
        svt_av1_compute_overall_elapsed_time_ms(start_time_seconds_tst,
                                                start_time_useconds_tst,
                                                finish_time_seconds_tst,
                                                finish_time_useconds_tst);

    svt_av1_get_time(&start_time_seconds_ref, &start_time_useconds_ref);

    for (int i = 0; i < num_loops; ++i)
        svt_av1_highbd_warp_affine_c(mat,
                                     input8b,
                                     input2b,
                                     w,
                                     h,
                                     stride8b,
                                     stride2b,
                                     output,
                                     32,
                                     32,
                                     out_w,
                                     out_h,
                                     out_w,
                                     sub_x,
                                     sub_y,
                                     bd,
                                     &conv_params,
                                     alpha,
                                     beta,
                                     gamma,
                                     delta);

    svt_av1_get_time(&finish_time_seconds_ref, &finish_time_useconds_ref);
    elapsed_time_ref =
        svt_av1_compute_overall_elapsed_time_ms(start_time_seconds_ref,
                                                start_time_useconds_ref,
                                                finish_time_seconds_ref,
                                                finish_time_useconds_ref);

    printf("highbd warp %3dx%-3d: %7.2fx faster\n",
           out_w,
           out_h,
           elapsed_time_ref / elapsed_time_tst);

    delete[] input8b_;
    delete[] input2b_;
    delete[] output;
    delete[] dsta;
}

void AV1HighbdWarpFilterTest::RunCheckOutput(
    highbd_warp_affine_func test_impl) {
    const int w = 128, h = 128;
    const int border = 16;
    const int stride8b = w + 2 * border;
    const int stride2b = w + 2 * border;
    HighbdWarpTestParam param = TEST_GET_PARAM(0);
    const int is_alpha_zero = TEST_GET_PARAM(1);
    const int is_beta_zero = TEST_GET_PARAM(2);
    const int is_gamma_zero = TEST_GET_PARAM(3);
    const int is_delta_zero = TEST_GET_PARAM(4);
    const int out_w = std::get<0>(param), out_h = std::get<1>(param);
    const int bd = std::get<3>(param);
    const int32_t ref = 0;
    const int num_iters = std::get<2>(param);
    const int mask = (1 << bd) - 1;
    int i, j, sub_x, sub_y;

    // The warp functions always write rows with widths that are multiples of 8.
    // So to avoid a buffer overflow, we may need to pad rows to a multiple
    // of 8.
    int output_n = ((out_w + 7) & ~7) * out_h;
    uint8_t *input8b_ = new uint8_t[h * stride8b];
    uint8_t *input8b = input8b_ + border;
    uint8_t *input2b_ = new uint8_t[h * stride2b];
    uint8_t *input2b = input2b_ + border;
    uint16_t *output = new uint16_t[output_n];
    uint16_t *output2 = new uint16_t[output_n];
    int32_t mat[8];
    int16_t alpha, beta, gamma, delta;
    ConvolveParams conv_params = get_conv_params(ref, 0, 0, bd);
    ConvBufType *dsta = new ConvBufType[output_n];
    ConvBufType *dstb = new ConvBufType[output_n];
    for (i = 0; i < output_n; ++i) {
        output[i] = output2[i] = rnd_->Rand16();
        dsta[i] = dstb[i] = 0;
    }

    for (i = 0; i < num_iters; ++i) {
        // Generate an input block and extend its borders horizontally
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c) {
                uint16_t val = rnd_->Rand16() & mask;
                input8b[r * stride8b + c] = val >> 2;
                input2b[r * stride2b + c] = (val & 3) << 6;
            }
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < border; ++c) {
                input8b[r * stride8b - border + c] = input8b[r * stride8b];
                input8b[r * stride8b + w + c] = input8b[r * stride8b + (w - 1)];
                input2b[r * stride2b - border + c] = input2b[r * stride2b];
                input2b[r * stride2b + w + c] = input2b[r * stride2b + (w - 1)];
            }
        }
        const int use_no_round = rnd_->Rand8() & 1;
        for (sub_x = 0; sub_x < 2; ++sub_x)
            for (sub_y = 0; sub_y < 2; ++sub_y) {
                generate_warped_model(rnd_,
                                      mat,
                                      &alpha,
                                      &beta,
                                      &gamma,
                                      &delta,
                                      is_alpha_zero,
                                      is_beta_zero,
                                      is_gamma_zero,
                                      is_delta_zero);
                for (int ii = 0; ii < 2; ++ii) {
                    for (int jj = 0; jj < 5; ++jj) {
                        for (int do_average = 0; do_average <= 1;
                             ++do_average) {
                            if (use_no_round) {
                                conv_params = get_conv_params_no_round(
                                    ref, do_average, 0, dsta, out_w, 1, bd);
                            } else {
                                conv_params = get_conv_params(ref, 0, 0, bd);
                            }
                            if (jj >= 4) {
                                conv_params.use_jnt_comp_avg = 0;
                            } else {
                                conv_params.use_jnt_comp_avg = 1;
                                conv_params.fwd_offset =
                                    quant_dist_lookup_table[ii][jj][0];
                                conv_params.bck_offset =
                                    quant_dist_lookup_table[ii][jj][1];
                            }

                            svt_av1_highbd_warp_affine_c(mat,
                                                         input8b,
                                                         input2b,
                                                         w,
                                                         h,
                                                         stride8b,
                                                         stride2b,
                                                         output,
                                                         32,
                                                         32,
                                                         out_w,
                                                         out_h,
                                                         out_w,
                                                         sub_x,
                                                         sub_y,
                                                         bd,
                                                         &conv_params,
                                                         alpha,
                                                         beta,
                                                         gamma,
                                                         delta);
                            if (use_no_round) {
                                // TODO(angiebird): Change this to test_impl
                                // once we have SIMD implementation
                                conv_params = get_conv_params_no_round(
                                    ref, do_average, 0, dstb, out_w, 1, bd);
                            }
                            if (jj >= 4) {
                                conv_params.use_jnt_comp_avg = 0;
                            } else {
                                conv_params.use_jnt_comp_avg = 1;
                                conv_params.fwd_offset =
                                    quant_dist_lookup_table[ii][jj][0];
                                conv_params.bck_offset =
                                    quant_dist_lookup_table[ii][jj][1];
                            }
                            test_impl(mat,
                                      input8b,
                                      input2b,
                                      w,
                                      h,
                                      stride8b,
                                      stride2b,
                                      output2,
                                      32,
                                      32,
                                      out_w,
                                      out_h,
                                      out_w,
                                      sub_x,
                                      sub_y,
                                      bd,
                                      &conv_params,
                                      alpha,
                                      beta,
                                      gamma,
                                      delta);

                            if (use_no_round) {
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(dsta[j], dstb[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(output[j], output2[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                            } else {
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(output[j], output2[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                            }
                        }
                    }
                }
            }
    }

    delete[] input8b_;
    delete[] input2b_;
    delete[] output;
    delete[] output2;
    delete[] dsta;
    delete[] dstb;
}
}  // namespace AV1HighbdWarpFilter
}  // namespace libaom_test
