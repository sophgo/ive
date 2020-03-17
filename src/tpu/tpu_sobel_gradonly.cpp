#include "../tpu_math/sqrt.hpp"
#include "tpu/tpu_sobel.hpp"

#include <string.h>

void IveTPUSobelGradOnly::setKernel(IveKernel &kernel_x, IveKernel &kernel_y) {
  m_kernel_x = &kernel_x;
  m_kernel_y = &kernel_y;
  m_kernel_info.size = m_kernel_x->img.m_tg.shape.h;
  int pad = (m_kernel_info.size - 1) / 2;
  m_kernel_info.pad[0] = pad;
  m_kernel_info.pad[1] = pad;
  m_kernel_info.pad[2] = pad;
  m_kernel_info.pad[3] = pad;
}

int IveTPUSobelGradOnly::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "sobelGrad";
  // 1 input tl
  // 2 conv result
  // 0 a^2 + b^2 result (reuse input tl)
  // 1 buf & 1 final sqrt result
  m_slice_info.io_fmt = FMT_BF16;
  m_slice_info.nums_of_tl = 3 * 2;   // in bf16
  m_kernel_info.nums_of_kernel = 4;  // 2 BF16 kernels
  return BM_SUCCESS;
}

int IveTPUSobelGradOnly::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                                  const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                                  const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                                  std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                                  const bool enable_cext) {
  bmk1880v2_tensor_lmem_shape_t tl_shape, tl_shape_out;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  tl_shape_out.n = tg_out_slices[0].n;
  tl_shape_out.c = tg_out_slices[0].c;
  tl_shape_out.h = tg_out_slices[0].h;
  tl_shape_out.w = tg_out_slices[0].w;
  auto *tl_input = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_gx = allocTLMem(bk_ctx, tl_shape_out, FMT_BF16, 1);
  auto *tl_gy = allocTLMem(bk_ctx, tl_shape_out, FMT_BF16, 1);

  bmk1880v2_tensor_lmem_shape_t tl_kernel_s = {1, m_kernel_x->img.m_tg.shape.c, m_kernel_info.size,
                                               m_kernel_info.size};
  auto *tl_kernel_gx = allocTLMem(bk_ctx, tl_kernel_s, FMT_BF16, 1, IVETLType::KERNEL);
  auto *tl_kernel_gy = allocTLMem(bk_ctx, tl_kernel_s, FMT_BF16, 1, IVETLType::KERNEL);
  cviImgFlush2TL(ctx, bk_ctx, m_kernel_x->img, tl_kernel_gx);
  cviImgFlush2TL(ctx, bk_ctx, m_kernel_y->img, tl_kernel_gy);

  if (enable_cext) {
    m_p_conv.pad_top = 0;
    m_p_conv.pad_bottom = 0;
  } else {
    m_p_conv.pad_top = m_kernel_info.pad[2];
    m_p_conv.pad_bottom = m_kernel_info.pad[3];
  }
  m_p_conv.pad_left = m_kernel_info.pad[0];
  m_p_conv.pad_right = m_kernel_info.pad[1];
  m_p_conv.stride_w = 1;
  m_p_conv.stride_h = 1;
  m_p_conv.relu_enable = 0;
  m_p_conv.ins_h = 0;
  m_p_conv.ins_w = 0;
  m_p_conv.ins_last_h = 0;
  m_p_conv.ins_last_w = 0;
  m_p_conv.dilation_h = 1;
  m_p_conv.dilation_w = 1;
  m_p_conv.bias = NULL;
  m_p_conv.rshift_bits = 0;
  m_p_conv.bf16_enable = 1;

  m_p_conv.ifmap = tl_input;

  tl_in_idx->push_back(0);
  tl_out_idx->push_back(1);
  tl_out_idx->push_back(2);
  return BM_SUCCESS;
}

void IveTPUSobelGradOnly::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  m_p_conv.ofmap = m_tl_vec[1];
  m_p_conv.weight = m_tl_vec[3];
  bmk1880v2_tiu_bf16_depthwise_convolution(bk_ctx, &m_p_conv);
  m_p_conv.ofmap = m_tl_vec[2];
  m_p_conv.weight = m_tl_vec[4];
  bmk1880v2_tiu_bf16_depthwise_convolution(bk_ctx, &m_p_conv);
}