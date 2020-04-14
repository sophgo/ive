#include "tpu/tpu_cmp.hpp"

#include <string.h>

void IveTPUMin::setKernelSize(u32 kz) {
  m_kernel_info.size = kz;
  m_kernel_info.pad[0] = 0;
  m_kernel_info.pad[1] = 0;
  m_kernel_info.pad[2] = 0;
  m_kernel_info.pad[3] = 0;
}

int IveTPUMin::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "min";
  m_slice_info.io_fmt = FMT_BF16;
  m_slice_info.nums_of_tl = 2 * 2;
  m_kernel_info.nums_of_kernel = 0;
  return BM_SUCCESS;
}

int IveTPUMin::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
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
  auto *tl_output = allocTLMem(bk_ctx, tl_shape_out, FMT_BF16, 1);

  m_p_mul.b_is_const = 1;
  m_p_mul.b_val = convert_fp32_bf16(-1.f);
  m_p_mul.bf16_enable = 1;
  m_p_mul.relu_enable = 0;
  m_p_mul.rshift_bits = 0;
  m_p_mul.res_high = NULL;
  m_p_mul.res_low = tl_input;
  m_p_mul.a = tl_input;

  m_p_mul_out.b_is_const = 1;
  m_p_mul_out.b_val = m_p_mul.b_val;
  m_p_mul_out.bf16_enable = 1;
  m_p_mul_out.relu_enable = 0;
  m_p_mul_out.rshift_bits = 0;
  m_p_mul_out.res_high = NULL;
  m_p_mul_out.res_low = tl_output;
  m_p_mul_out.a = tl_output;

  m_p_max.ifmap = tl_input;
  m_p_max.ofmap = tl_output;
  m_p_max.kh = m_kernel_info.size;
  m_p_max.kw = m_kernel_info.size;
  m_p_max.pad_top = 0;
  m_p_max.pad_bottom = 0;
  m_p_max.pad_left = 0;
  m_p_max.pad_right = 0;
  m_p_max.stride_w = 1;
  m_p_max.stride_h = 1;
  m_p_max.bf16_enable = 1;
  m_p_max.ins_fp = 0;  // Useless in max pooling but we still give it a value.

  tl_in_idx->push_back(0);
  tl_out_idx->push_back(1);
  return BM_SUCCESS;
}

void IveTPUMin::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  bmk1880v2_tiu_bf16_element_wise_mul(bk_ctx, &m_p_mul);
  bmk1880v2_tiu_bf16_max_pooling(bk_ctx, &m_p_max);
  bmk1880v2_tiu_bf16_element_wise_mul(bk_ctx, &m_p_mul_out);
}