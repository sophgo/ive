#include "tpu/tpu_threshold.hpp"
#include "utils.hpp"

#include <string.h>

void IveTPUThresholdSlope::setThreshold(int low, int high) {
  m_threshold_low = low;
  m_threshold_high = high;
}

int IveTPUThresholdSlope::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "threshSlope";
  m_slice_info.nums_of_tl = 1;

  return BM_SUCCESS;
}

int IveTPUThresholdSlope::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
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

  m_p_max.a = tl_input;
  m_p_max.b = NULL;
  m_p_max.max = tl_input;
  m_p_max.bf16_enable = 0;
  m_p_max.b_is_const = 1;
  m_p_max.b_is_signed = 0;
  m_p_max.b_val = m_threshold_low;

  m_p_min.a = tl_input;
  m_p_min.b = NULL;
  m_p_min.min = tl_input;
  m_p_min.bf16_enable = 0;
  m_p_min.b_is_const = 1;
  m_p_min.b_is_signed = 0;
  m_p_min.b_val = m_threshold_high;

  tl_in_idx->push_back(0);
  tl_out_idx->push_back(0);
  return BM_SUCCESS;
}

void IveTPUThresholdSlope::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  bmk1880v2_tiu_element_wise_max(bk_ctx, &m_p_max);
  bmk1880v2_tiu_element_wise_min(bk_ctx, &m_p_min);
}