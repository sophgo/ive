#include "tpu/tpu_threshold.hpp"
#include "utils.hpp"

#include <string.h>

void IveTPUThresholdHighLow::setThreshold(int threshold, int low, int high) {
  m_threshold = threshold;
  m_threshold_low = low;
  m_threshold_high = high;
}

int IveTPUThresholdHighLow::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "threshHL";
  m_slice_info.nums_of_tl = 3;

  return CVI_SUCCESS;
}

int IveTPUThresholdHighLow::runSetup(
    bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
    const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
    const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices, std::vector<u32> *tl_in_idx,
    std::vector<u32> *tl_out_idx, const bool enable_cext) {
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
  m_p_mul.res_low = tl_input;
  m_p_mul.rshift_bits = 0;

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
  return CVI_SUCCESS;
}

void IveTPUThresholdHighLow::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  bmk1880v2_tiu_element_wise_mac(bk_ctx, &m_p_mac);
  bmk1880v2_tiu_element_wise_mul(bk_ctx, &m_p_mul);
  bmk1880v2_tiu_element_wise_max(bk_ctx, &m_p_max);
  bmk1880v2_tiu_element_wise_min(bk_ctx, &m_p_min);
}