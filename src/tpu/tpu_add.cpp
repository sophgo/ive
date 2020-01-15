#include "tpu/tpu_add.hpp"
#include <string.h>

int IveTPUAdd::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_slice_info.nums_of_tl = 3;

  return BM_SUCCESS;
}

int IveTPUAdd::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
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
  auto *high_bit_zeros = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);

  extendValue2TL(ctx, bk_ctx, 0, tl_shape.n * tl_shape.c, tl_shape.h, tl_shape.w, FMT_U8,
                 high_bit_zeros);

  m_p_add.res_high = NULL;
  m_p_add.res_low = tl_input;
  m_p_add.a_high = high_bit_zeros;
  m_p_add.a_low = tl_input;
  m_p_add.b_is_const = 0;
  m_p_add.b_high = high_bit_zeros;
  m_p_add.b_low = tl_input2;
  m_p_add.rshift_bits = 0;
  m_p_add.relu_enable = 0;

  tl_in_idx->push_back(0);
  tl_in_idx->push_back(1);
  tl_out_idx->push_back(0);
  return BM_SUCCESS;
}

void IveTPUAdd::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  bmk1880v2_tiu_element_wise_add(bk_ctx, &m_p_add);
}