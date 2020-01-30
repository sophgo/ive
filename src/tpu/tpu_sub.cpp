#include "tpu/tpu_sub.hpp"
#include "utils.hpp"

#include <string.h>

int IveTPUSub::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_slice_info.nums_of_tl = 3;

  return BM_SUCCESS;
}

int IveTPUSub::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
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
  auto *tl_high_bit = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);

  extendValue2TL(ctx, bk_ctx, 0, tl_shape.n * tl_shape.c, tl_shape.h, tl_shape.w, FMT_U8,
                 tl_high_bit);

  m_p_mac.res_high = tl_high_bit;
  m_p_mac.res_low = tl_input;
  m_p_mac.a = tl_input2;
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

void IveTPUSub::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  bmk1880v2_tiu_element_wise_mac(bk_ctx, &m_p_mac);
}