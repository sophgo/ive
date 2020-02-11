#include "tpu/tpu_magandang.hpp"

#include <string.h>
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"

int IveTPUMadAndAng::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  // 2 input tl
  //  tmp buf
  // 1 atan & 1 final sqrt result
  m_slice_info.nums_of_tl = 10 * 2;                // in bf16
  m_slice_info.table_size_per_channel = 512 * 9;  // 8 table in bf16
  m_kernel_info.nums_of_kernel = 0;               // 2 BF16 kernels
  return BM_SUCCESS;
}

int IveTPUMadAndAng::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                              const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                              const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                              std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) {
  bmk1880v2_tensor_lmem_shape_t tl_shape;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  auto *tl_input = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_input2 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_mag = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_angle = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_buf = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_buf2 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_buf3 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_buf4 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_buf5 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_buf6 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);

  bmk1880v2_tensor_lmem_shape_t tl_table_s;
  bf16_lut_tbl_bytesize(bk_ctx, &tl_table_s, FMT_BF16);  // 32 * 8
  // atan buf
  auto *tl_y0_buf = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  auto *tl_slope_buf = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  auto *tl_invert_buf = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  auto *tl_pos_neg_buf = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  {
    CviImg table_data_atan_y0(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
    CviImg table_data_atan_slope(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
    CviImg table_data_atan_invert(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
    CviImg table_data_atan_pos_neg(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
    bf16_atan_tbl((u16 *)table_data_atan_y0.GetVAddr(), (u16 *)table_data_atan_slope.GetVAddr(),
                  (u16 *)table_data_atan_invert.GetVAddr(),
                  (u16 *)table_data_atan_pos_neg.GetVAddr(), &tl_table_s);
    bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
    p.src = &table_data_atan_y0.m_tg;
    p.dst = tl_y0_buf;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    p.src = &table_data_atan_slope.m_tg;
    p.dst = tl_slope_buf;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    p.src = &table_data_atan_invert.m_tg;
    p.dst = tl_invert_buf;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    p.src = &table_data_atan_pos_neg.m_tg;
    p.dst = tl_pos_neg_buf;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    bmruntime_bmkernel_submit(*ctx);
    table_data_atan_y0.Free(ctx);
    table_data_atan_slope.Free(ctx);
    table_data_atan_invert.Free(ctx);
    table_data_atan_pos_neg.Free(ctx);
  }
  auto *tl_reciprocal_table_answer = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  auto *tl_reciprocal_table_answer_mantissa = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  {
    CviImg table_data(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
    CviImg table_data_mantissa(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
    bf16_reciprocal_tbl((u16 *)table_data.GetVAddr(), (u16 *)table_data_mantissa.GetVAddr(),
                        &tl_table_s);
    bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
    p.src = &table_data.m_tg;
    p.dst = tl_reciprocal_table_answer;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    p.src = &table_data_mantissa.m_tg;
    p.dst = tl_reciprocal_table_answer_mantissa;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    bmruntime_bmkernel_submit(*ctx);
    table_data.Free(ctx);
    table_data_mantissa.Free(ctx);
  }
  auto *tl_sqrt_table_answer = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  auto *tl_sqrt_table_answer_mantissa = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  {
    CviImg table_data(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
    CviImg table_data_mantissa(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
    bf16_sqrt_tbl((u16 *)table_data.GetVAddr(), (u16 *)table_data_mantissa.GetVAddr(), &tl_table_s);
    bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
    p.src = &table_data.m_tg;
    p.dst = tl_sqrt_table_answer;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    p.src = &table_data_mantissa.m_tg;
    p.dst = tl_sqrt_table_answer_mantissa;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    bmruntime_bmkernel_submit(*ctx);
    table_data.Free(ctx);
    table_data_mantissa.Free(ctx);
  }

  auto *tl_idx_0_table = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  {
    CviImg idx_0_table_data(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
    bf16_gen_0_tbl((u16 *)idx_0_table_data.GetVAddr(), &tl_table_s);
    bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
    p.src = &idx_0_table_data.m_tg;
    p.dst = tl_idx_0_table;
    bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
    bmruntime_bmkernel_submit(*ctx);
    idx_0_table_data.Free(ctx);
  }

  m_p_mul.rshift_bits = 0;
  m_p_mul.relu_enable = 1;
  m_p_mul.bf16_enable = 1;

  m_p_mac.res_is_int8 = 0;
  m_p_mac.lshift_bits = 0;
  m_p_mac.rshift_bits = 0;
  m_p_mac.relu_enable = 1;
  m_p_mac.bf16_enable = 1;

  m_p_mul.res_high = NULL;
  m_p_mul.res_low = tl_buf;
  m_p_mul.a = tl_input;
  m_p_mul.b_is_const = 0;
  m_p_mul.b = tl_input;
  m_p_mac.res_high = NULL;
  m_p_mac.res_low = tl_buf;
  m_p_mac.a = tl_input2;
  m_p_mac.b_is_const = 0;
  m_p_mac.b = tl_input2;

  m_p_sqrt.a = tl_buf;
  m_p_sqrt.res = tl_mag;
  m_p_sqrt.buf = tl_buf2;
  m_p_sqrt.sqrt_table_answer = tl_sqrt_table_answer;
  m_p_sqrt.sqrt_table_answer_mantissa = tl_sqrt_table_answer_mantissa;

  m_p_atan2.a = tl_input;
  m_p_atan2.b = tl_input2;
  m_p_atan2.res = tl_angle;
  m_p_atan2.buf1 = tl_buf;
  m_p_atan2.buf2 = tl_buf2;
  m_p_atan2.buf3 = tl_buf3;
  m_p_atan2.buf4 = tl_buf4;
  m_p_atan2.buf5 = tl_buf5;
  m_p_atan2.buf6 = tl_buf6;
  m_p_atan2.y0 = tl_y0_buf;
  m_p_atan2.slope = tl_slope_buf;
  m_p_atan2.invert = tl_invert_buf;
  m_p_atan2.pos_neg = tl_pos_neg_buf;
  m_p_atan2.reciprocal_table_answer = tl_reciprocal_table_answer;
  m_p_atan2.reciprocal_table_answer_mantissa = tl_reciprocal_table_answer_mantissa;
  m_p_atan2.sqrt_table_answer = tl_sqrt_table_answer;
  m_p_atan2.sqrt_table_answer_mantissa = tl_sqrt_table_answer_mantissa;
  m_p_atan2.idx_0_table = tl_idx_0_table;
  m_p_atan2.fmt = FMT_BF16;

  m_p_mul_const.rshift_bits = 0;
  m_p_mul_const.relu_enable = 0;
  m_p_mul_const.bf16_enable = 1;
  m_p_mul_const.res_high = NULL;
  m_p_mul_const.res_low = tl_angle;
  m_p_mul_const.a = tl_angle;
  m_p_mul_const.b_is_const = 1;
  m_p_mul_const.b_val = convert_fp32_bf16(180.f / M_PI);

  tl_in_idx->push_back(0);
  tl_in_idx->push_back(1);
  tl_out_idx->push_back(2);
  tl_out_idx->push_back(3);
  return BM_SUCCESS;
}
#include "../tpu_math/1880v2_utils.h"
void IveTPUMadAndAng::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  bmk1880v2_tiu_bf16_element_wise_mul(bk_ctx, &m_p_mul);
  bmk1880v2_tiu_bf16_element_wise_mac(bk_ctx, &m_p_mac);
  bf16_emit_sqrt(bk_ctx, m_p_sqrt.a, m_p_sqrt.buf, m_p_sqrt.sqrt_table_answer,
                 m_p_sqrt.sqrt_table_answer_mantissa, m_p_sqrt.res);
  // sqrt goes first cause atan2 changes y value
  bf16_atan2_emit(bk_ctx, m_p_atan2.b, m_p_atan2.a, m_p_atan2.buf1, m_p_atan2.buf2, m_p_atan2.buf3,
                  m_p_atan2.buf4, m_p_atan2.buf5, m_p_atan2.buf6, m_p_atan2.y0, m_p_atan2.slope,
                  m_p_atan2.invert, m_p_atan2.pos_neg, m_p_atan2.reciprocal_table_answer,
                  m_p_atan2.reciprocal_table_answer_mantissa, m_p_atan2.sqrt_table_answer,
                  m_p_atan2.sqrt_table_answer_mantissa, m_p_atan2.idx_0_table, m_p_atan2.res,
                  m_p_atan2.fmt);
  bmk1880v2_tiu_bf16_element_wise_mul(bk_ctx, &m_p_mul_const);
}