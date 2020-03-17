#include <string.h>
#include "tpu/tpu_copy.hpp"

void IveTPUCopyInterval::setInvertal(u32 hori, u32 verti) {
  m_hori = hori;
  m_verti = verti;
}

int IveTPUCopyInterval::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "intCopy";
  m_slice_info.nums_of_tl = 1 + m_hori * m_verti;

  return BM_SUCCESS;
}

int IveTPUCopyInterval::sliceSetup(SliceRes &slice_res, SliceRes *tg_in_res, SliceRes *tg_out_res) {
  *tg_in_res = slice_res;
  *tg_out_res = slice_res;
  tg_out_res->h.skip *= m_verti;
  tg_out_res->w.skip *= m_hori;
  tg_out_res->h.left *= m_verti;
  tg_out_res->h.slice *= m_verti;
  tg_out_res->w.slice *= m_hori;
  tg_out_res->w.left *= m_hori;
  return BM_SUCCESS;
}

int IveTPUCopyInterval::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
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
  bmk1880v2_tensor_lmem_shape_t tl_shape2;
  tl_shape2.n = tg_out_slices[0].n;
  tl_shape2.c = tg_out_slices[0].c;
  tl_shape2.h = tg_out_slices[0].h;
  tl_shape2.w = tg_out_slices[0].w;
  auto *tl_output = allocTLMem(bk_ctx, tl_shape2, FMT_U8, 1);
  m_p_copy.bf16_enable = tl_input->fmt == FMT_BF16 ? 1 : 0;
  m_p_copy.src = tl_input;
  m_p_copy.dst = tl_output;

  bmk1880v2_tdma_tg2l_tensor_fill_constant_param_t p_fill;
  p_fill.constant = 0;
  p_fill.dst = tl_output;
  bmk1880v2_tdma_tg2l_bf16_tensor_fill_constant(bk_ctx, &p_fill);
  bmruntime_bmkernel_submit(*ctx);

  tl_in_idx->push_back(0);
  tl_out_idx->push_back(1);
  return BM_SUCCESS;
}

void IveTPUCopyInterval::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  auto *tl_in = m_tl_vec[0];
  auto *tl_out = m_tl_vec[1];
  auto shape = tl_out->shape;
  auto stride = tl_out->stride;
  // Change shape and stride for interval hack.
  tl_out->shape = tl_in->shape;
  tl_out->stride.w *= m_hori;
  tl_out->stride.h *= m_verti;
  bmk1880v2_tiu_bf16_element_wise_copy(bk_ctx, &m_p_copy);
  // Change to default value.
  tl_out->shape = shape;
  tl_out->stride = stride;
}