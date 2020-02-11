#include "tpu/tpu_sobel.hpp"

#include <string.h>

void IveTPUSobel::setKernel(const IveKernel &kernel_x, const IveKernel &kernel_y) {
  m_kernel_x = &kernel_x;
  m_kernel_y = &kernel_y;
  m_kernel_info.size = m_kernel_x->img.m_tg.shape.h;
  int pad = (m_kernel_info.size - 1) / 2;
  m_kernel_info.pad[0] = pad;
  m_kernel_info.pad[1] = pad;
  m_kernel_info.pad[2] = pad;
  m_kernel_info.pad[3] = pad;
}

int IveTPUSobel::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  // 1 input tl
  // 2 conv result
  // 0 a^2 + b^2 result (reuse input tl)
  // 1 buf & 1 final sqrt result
  m_slice_info.nums_of_tl = 4 * 2;             // in bf16
  m_slice_info.table_size_per_channel = 1024;  // sqrt 2 table 256 * 2 in bf16
  m_kernel_info.nums_of_kernel = 4;            // 2 BF16 kernels
  return BM_SUCCESS;
}

int IveTPUSobel::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                          const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                          const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                          std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) {
  bmk1880v2_tensor_lmem_shape_t tl_shape;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  auto *tl_input = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_gx = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_gy = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_res = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);

  bmk1880v2_tensor_lmem_shape_t tl_kernel_s = {1, tl_shape.c, m_kernel_info.size,
                                               m_kernel_info.size};
  auto *tl_kernel_gx = allocTLMem(bk_ctx, tl_kernel_s, FMT_BF16, 1);
  {
    bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
    p.src = &m_kernel_x->img.m_tg;
    p.dst = tl_kernel_gx;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    bmruntime_bmkernel_submit(*ctx);
  }
  auto *tl_kernel_gy = allocTLMem(bk_ctx, tl_kernel_s, FMT_BF16, 1);
  {
    bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
    p.src = &m_kernel_y->img.m_tg;
    p.dst = tl_kernel_gy;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    bmruntime_bmkernel_submit(*ctx);
  }
  bmk1880v2_tensor_lmem_shape_t tl_table_s = {1, 32, 32, 8};
  auto *tl_table_data = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  auto *tl_table_data_mantissa = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  {
    CviImg table_data(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
    CviImg table_data_mantissa(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
    bf16_sqrt_tbl((u16 *)table_data.GetVAddr(), (u16 *)table_data_mantissa.GetVAddr(), &tl_table_s);
    bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
    p.src = &table_data.m_tg;
    p.dst = tl_table_data;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    p.src = &table_data_mantissa.m_tg;
    p.dst = tl_table_data_mantissa;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    bmruntime_bmkernel_submit(*ctx);
    table_data.Free(ctx);
    table_data_mantissa.Free(ctx);
  }

  m_p_conv.pad_top = m_kernel_info.pad[2];
  m_p_conv.pad_bottom = m_kernel_info.pad[3];
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

  m_p_mul.rshift_bits = 0;
  m_p_mul.relu_enable = 1;
  m_p_mul.bf16_enable = 1;

  m_p_mac.res_is_int8 = 0;
  m_p_mac.lshift_bits = 0;
  m_p_mac.rshift_bits = 0;
  m_p_mac.relu_enable = 1;
  m_p_mac.bf16_enable = 1;

  m_p_conv.ifmap = tl_input;
  m_p_mul.res_high = NULL;
  m_p_mul.res_low = tl_res;
  m_p_mul.a = tl_gx;
  m_p_mul.b_is_const = 0;
  m_p_mul.b = tl_gx;
  m_p_mac.res_high = NULL;
  m_p_mac.res_low = tl_res;
  m_p_mac.a = tl_gy;
  m_p_mac.b_is_const = 0;
  m_p_mac.b = tl_gy;

  m_p_sqrt.a = tl_res;
  m_p_sqrt.res = tl_gx;
  m_p_sqrt.buf = tl_input;
  m_p_sqrt.sqrt_table_answer = tl_table_data;
  m_p_sqrt.sqrt_table_answer_mantissa = tl_table_data_mantissa;

  tl_in_idx->push_back(0);
  tl_out_idx->push_back(1);
  return BM_SUCCESS;
}

void IveTPUSobel::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_p_conv.ofmap = m_tl_vec[1];
  m_p_conv.weight = m_tl_vec[4];
  bmk1880v2_tiu_bf16_depthwise_convolution(bk_ctx, &m_p_conv);
  bmruntime_bmkernel_submit(*ctx);
  m_p_conv.ofmap = m_tl_vec[2];
  m_p_conv.weight = m_tl_vec[5];
  bmk1880v2_tiu_bf16_depthwise_convolution(bk_ctx, &m_p_conv);
  bmk1880v2_tiu_bf16_element_wise_mul(bk_ctx, &m_p_mul);
  bmk1880v2_tiu_bf16_element_wise_mac(bk_ctx, &m_p_mac);
  bf16_emit_sqrt(bk_ctx, m_p_sqrt.a, m_p_sqrt.buf, m_p_sqrt.sqrt_table_answer,
                 m_p_sqrt.sqrt_table_answer_mantissa, m_p_sqrt.res);
}