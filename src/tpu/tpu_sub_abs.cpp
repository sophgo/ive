#include "tpu/tpu_sub.hpp"
#include "utils.hpp"

#include <string.h>

int IveTPUSubAbs::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  // 2 - in
  // 1 -tmp
  // 1 -high bit
  m_slice_info.nums_of_tl = 4;

  return BM_SUCCESS;
}

int IveTPUSubAbs::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                           const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                           const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                           std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) {
  bmk1880v2_tensor_lmem_shape_t tl_shape;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  auto *tl_input = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);
  auto *tl_input2 = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);
  auto *tl_min = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);
  auto *tl_high_bit = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);

  m_p_min.a = tl_input;
  m_p_min.b = tl_input2;
  m_p_min.b_is_const = 0;
  m_p_min.bf16_enable = 0;
  m_p_min.min = tl_min;

  m_p_max.a = tl_input;
  m_p_max.b = tl_input2;
  m_p_max.b_is_const = 0;
  m_p_max.bf16_enable = 0;
  m_p_max.max = tl_input;

  m_p_mac.res_high = tl_high_bit;
  m_p_mac.res_low = tl_input;
  m_p_mac.a = tl_min;
  m_p_mac.b_is_const = 1;
  m_p_mac.b_val = -1;
  m_p_mac.b_is_signed = 1;
  m_p_mac.lshift_bits = 0;
  m_p_mac.res_is_int8 = 1;
  m_p_mac.rshift_bits = 0;
  m_p_mac.relu_enable = 1;

  tl_in_idx->push_back(0);
  tl_in_idx->push_back(1);
  tl_out_idx->push_back(0);
  return BM_SUCCESS;
}

void IveTPUSubAbs::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  bmk1880v2_tiu_element_wise_min(bk_ctx, &m_p_min);
  bmk1880v2_tiu_element_wise_max(bk_ctx, &m_p_max);
  bmk1880v2_tiu_element_wise_mac(bk_ctx, &m_p_mac);
}