#include "tpu/tpu_sub.hpp"
#include "utils.hpp"

#include <string.h>

int IveTPUSub::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "sub";
  m_slice_info.ping_pong_size = 2;
  m_slice_info.ping_pong_share_tl = 1;
  m_slice_info.nums_of_tl = 3;

  return BM_SUCCESS;
}

int IveTPUSub::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                        const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                        const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                        std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                        const bool enable_cext) {
  m_input1.clear();
  m_input2.clear();
  bmk1880v2_tensor_lmem_shape_t tl_shape;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  for (size_t i = 0; i < m_slice_info.ping_pong_size; i++) {
    m_input1.emplace_back(allocTLMem(bk_ctx, tl_shape, FMT_U8, 1));
    m_input2.emplace_back(allocTLMem(bk_ctx, tl_shape, FMT_U8, 1));
  }
  auto *tl_high_bit = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);

  constantFillTL(ctx, bk_ctx, 0, tl_high_bit);

  m_p_mac.res_high = tl_high_bit;
  m_p_mac.b_is_const = 1;
  m_p_mac.b_val = -1;
  m_p_mac.b_is_signed = 1;
  m_p_mac.lshift_bits = 0;
  m_p_mac.res_is_int8 = 1;
  m_p_mac.rshift_bits = 0;
  m_p_mac.relu_enable = 1;

  for (size_t pp = 0; pp < m_slice_info.ping_pong_size; pp++) {
    tl_in_idx->push_back(0 + pp * 2);
    tl_in_idx->push_back(1 + pp * 2);
    tl_out_idx->push_back(0 + pp * 2);
  }
  return BM_SUCCESS;
}

void IveTPUSub::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  m_p_mac.res_low = m_input1[ping_idx];
  m_p_mac.a = m_input2[ping_idx];
  bmk1880v2_tiu_element_wise_mac(bk_ctx, &m_p_mac);
}