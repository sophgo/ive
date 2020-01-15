#include "tpu/tpu_and.hpp"
#include <string.h>

int IveTPUAnd::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_slice_info.nums_of_tl = 2;

  return BM_SUCCESS;
}

int IveTPUAnd::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
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

  m_p_and.a = tl_input;
  m_p_and.b = tl_input2;
  m_p_and.res = tl_input;
  m_p_and.layer_id = 0;

  tl_in_idx->push_back(0);
  tl_in_idx->push_back(1);
  tl_out_idx->push_back(0);
  return BM_SUCCESS;
}

void IveTPUAnd::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  bmk1880v2_tiu_element_wise_and_int8(bk_ctx, &m_p_and);
}