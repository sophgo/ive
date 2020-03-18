#include "tpu/tpu_normalize.hpp"
#include <string.h>

void IveTPUNormalize::setMinMax(float min, float max) {
  m_min = min;
  m_max = max;
}

void IveTPUNormalize::setOutputFMT(fmt_t fmt) { m_fmt = fmt; }

int IveTPUNormalize::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "normalize";
  m_slice_info.ping_pong_size = 2;
  m_slice_info.nums_of_tl = 1 * 2;

  return BM_SUCCESS;
}

int IveTPUNormalize::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                              const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                              const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                              std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                              const bool enable_cext) {
  m_input.clear();
  if (m_fmt != FMT_U8 && m_fmt != FMT_I8) {
    std::cerr << "TPUT normalize only supports U8/ I8." << std::endl;
  }
  bmk1880v2_tensor_lmem_shape_t tl_shape;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  for (size_t i = 0; i < m_slice_info.ping_pong_size; i++) {
    m_input.emplace_back(allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1));
  }

  m_p_add.res_high = NULL;
  m_p_add.a_high = NULL;
  m_p_add.b_is_const = 1;
  m_p_add.b_val = convert_fp32_bf16(-1.f * m_min);
  m_p_add.rshift_bits = 0;
  m_p_add.relu_enable = 0;
  m_p_add.bf16_enable = 1;

  m_p_mul.res_high = NULL;
  m_p_mul.b_is_const = 1;
  m_p_mul.b_val = convert_fp32_bf16(255.f / (float)(m_max - m_min));
  m_p_mul.rshift_bits = 0;
  m_p_mul.relu_enable = 0;
  m_p_mul.bf16_enable = 1;

  if (m_fmt == FMT_I8) {
    m_p_add_offset.res_high = NULL;
    m_p_add_offset.a_high = NULL;
    m_p_add_offset.b_is_const = 1;
    m_p_add_offset.b_val = convert_fp32_bf16(-128.f);
    m_p_add_offset.rshift_bits = 0;
    m_p_add_offset.relu_enable = 0;
    m_p_add_offset.bf16_enable = 1;
  }

  for (size_t pp = 0; pp < m_slice_info.ping_pong_size; pp++) {
    tl_in_idx->push_back(0 + pp);
    tl_out_idx->push_back(0 + pp);
  }
  return BM_SUCCESS;
}

void IveTPUNormalize::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  m_p_add.res_low = m_input[ping_idx];
  m_p_add.a_low = m_input[ping_idx];
  m_p_mul.res_low = m_input[ping_idx];
  m_p_mul.a = m_input[ping_idx];
  bmk1880v2_tiu_bf16_element_wise_add(bk_ctx, &m_p_add);
  bmk1880v2_tiu_bf16_element_wise_mul(bk_ctx, &m_p_mul);
  if (m_fmt == FMT_I8) {
    m_p_add_offset.res_low = m_input[ping_idx];
    m_p_add_offset.a_low = m_input[ping_idx];
    bmk1880v2_tiu_bf16_element_wise_add(bk_ctx, &m_p_add_offset);
  }
}