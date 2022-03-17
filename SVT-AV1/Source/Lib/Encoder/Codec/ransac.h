/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#ifndef AOM_AV1_ENCODER_RANSAC_H_
#define AOM_AV1_ENCODER_RANSAC_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>

#include "global_motion.h"

typedef int (*RansacFunc)(int *matched_points, int npoints, int *num_inliers_by_motion,
                          MotionModel *params_by_motion, int num_motions);
RansacFunc svt_av1_get_ransac_type(TransformationType type);
#endif // AOM_AV1_ENCODER_RANSAC_H_
