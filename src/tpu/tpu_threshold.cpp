#include "tpu/tpu_threshold.hpp"
#include "utils.hpp"

#include <string.h>

void IveTPUThreshold::setThreshold(int threshold) { m_threshold = threshold; }

int IveTPUThreshold::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_slice_info.nums_of_tl = 3;

  return BM_SUCCESS;
}

int IveTPUThreshold::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                              const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                              const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                              std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                              const bool enable_cext) {
  bmk1880v2_tensor_lmem_shape_t tl_shape;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  auto *tl_input = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);
  auto *tl_threshold = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);
  auto *tl_high_bit = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);

  if (m_threshold == -1) {
    std::cerr << "threshold not set." << std::endl;
  }
  if (m_threshold == 0) {
    m_threshold = 1;
  }

  constantFillTL(ctx, bk_ctx, m_threshold - 1, tl_threshold);
  constantFillTL(ctx, bk_ctx, 0, tl_high_bit);

  m_p_mac.res_high = tl_high_bit;
  m_p_mac.res_low = tl_input;
  m_p_mac.a = tl_threshold;
  m_p_mac.b_is_const = 1;
  m_p_mac.b_val = -1;
  m_p_mac.b_is_signed = 1;
  m_p_mac.lshift_bits = 0;
  m_p_mac.res_is_int8 = 1;
  m_p_mac.rshift_bits = 0;
  m_p_mac.relu_enable = 1;

  m_p_mul.a = tl_input;
  m_p_mul.b_val = 255;
  m_p_mul.b_is_const = 1;
  m_p_mul.b_is_signed = 0;
  m_p_mul.bf16_enable = 0;
  m_p_mul.relu_enable = 0;
  m_p_mul.res_high = NULL;
  m_p_mul.res_low = m_tl_vec[0];
  m_p_mul.rshift_bits = 0;

  tl_in_idx->push_back(0);
  tl_out_idx->push_back(0);
  return BM_SUCCESS;
}

void IveTPUThreshold::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  bmk1880v2_tiu_element_wise_mac(bk_ctx, &m_p_mac);
  bmk1880v2_tiu_element_wise_mul(bk_ctx, &m_p_mul);
}