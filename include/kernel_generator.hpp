#pragma once
#include "tpu_data.hpp"

enum IVE_KERNEL {
  GAUSSIAN = 0,
  SOBEL_X,
  SOBEL_Y,
  SCHARR_X,
  SCHARR_Y,
  MORPH_RECT,
  MORPH_CROSS,
  MORPH_ELLIPSE,
  CUSTOM
};

IveKernel createKernel(bmctx_t *ctx, u32 img_c, u32 k_h, u32 k_w, IVE_KERNEL type);