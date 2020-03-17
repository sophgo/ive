#include "tpu/tpu_magandang.hpp"

#include <string.h>
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#ifdef MAGnANG_DEBUG
#include "../tpu_math/1880v2_utils.h"
#endif

void IveTPUMagAndAng::setTblMgr(TblMgr *tblmgr) { mp_tblmgr = tblmgr; }

void IveTPUMagAndAng::exportOption(bool mag_value, bool ang_value) {
  m_export_mag = mag_value;
  m_export_ang = ang_value;
}

void IveTPUMagAndAng::noNegative(bool value) { m_no_negative = value; }

int IveTPUMagAndAng::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "magnang";
  // 2 input tl
  // 6 tmp buf
  // 1 atan & 1 final sqrt result
  m_slice_info.nums_of_tl = 10 * 2;  // in bf16
  m_slice_info.nums_of_table = 9 * 2;
  m_kernel_info.nums_of_kernel = 0;  // 2 BF16 kernels

  return BM_SUCCESS;
}

int IveTPUMagAndAng::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                              const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                              const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                              std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                              const bool enable_cext) {
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
  // FIXME: Dirty workaround. Allocate a 1x1 tg also fixed in cmodel.
  // May not work in soc mode.
  // CviImg workaround_img(ctx, 1, 1, 1, FMT_BF16);
  auto *tl_buf2 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_buf3 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_buf4 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_buf5 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_buf6 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);

  const bmk1880v2_tensor_lmem_shape_t tl_table_s = mp_tblmgr->getTblTLShape();
  // atan buf
  auto *tl_y0_table = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  auto *tl_slope_table = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  auto *tl_invert_table = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  auto *tl_pos_neg_table = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  {
    const CviImg *table_data_atan_y0 = mp_tblmgr->atan(TBLATAN::TBLATAN_Y0);
    const CviImg *table_data_atan_slope = mp_tblmgr->atan(TBLATAN::TBLATAN_SLOPE);
    const CviImg *table_data_atan_invert = mp_tblmgr->atan(TBLATAN::TBLATAN_INVERT);
    const CviImg *table_data_atan_pos_neg = mp_tblmgr->atan(TBLATAN::TBLATAN_POSNEG);
    cviImg2TL(ctx, bk_ctx, *table_data_atan_y0, tl_y0_table);
    cviImg2TL(ctx, bk_ctx, *table_data_atan_slope, tl_slope_table);
    cviImg2TL(ctx, bk_ctx, *table_data_atan_invert, tl_invert_table);
    cviImg2TL(ctx, bk_ctx, *table_data_atan_pos_neg, tl_pos_neg_table);
  }
  auto *tl_reciprocal_table_answer = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  auto *tl_reciprocal_table_answer_mantissa = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  {
    const CviImg *table_data = mp_tblmgr->reciprocal(TBLRECIPROCAL::TBLRECIPROCAL_DATA);
    const CviImg *table_data_mantissa =
        mp_tblmgr->reciprocal(TBLRECIPROCAL::TBLRECIPROCAL_MANTISSA);
    cviImg2TL(ctx, bk_ctx, *table_data, tl_reciprocal_table_answer);
    cviImg2TL(ctx, bk_ctx, *table_data_mantissa, tl_reciprocal_table_answer_mantissa);
  }
  auto *tl_sqrt_table_answer = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  auto *tl_sqrt_table_answer_mantissa = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  {
    const CviImg *table_data = mp_tblmgr->sqrt(TBLSQRT::TBLSQRT_DATA);
    const CviImg *table_data_mantissa = mp_tblmgr->sqrt(TBLSQRT::TBLSQRT_MANTISSA);
    cviImg2TL(ctx, bk_ctx, *table_data, tl_sqrt_table_answer);
    cviImg2TL(ctx, bk_ctx, *table_data_mantissa, tl_sqrt_table_answer_mantissa);
  }

  auto *tl_idx_0_table = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
  {
    const CviImg *idx_0_table_data = mp_tblmgr->mask(TBLMASK::TBLMASK_ZERO);
    cviImg2TL(ctx, bk_ctx, *idx_0_table_data, tl_idx_0_table);
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
  /** FIXME: Dirty workaround. tl polution.
   *  m_p_atan2.buf2 = tl_buf2;
   *  m_p_atan2.buf3 = tl_buf3;
   */
  m_p_atan2.buf2 = tl_buf2;
  m_p_atan2.buf3 = tl_buf3;
  m_p_atan2.buf4 = tl_buf4;
  m_p_atan2.buf5 = tl_buf5;
  m_p_atan2.buf6 = tl_buf6;
  m_p_atan2.y0 = tl_y0_table;
  m_p_atan2.slope = tl_slope_table;
  m_p_atan2.invert = tl_invert_table;
  m_p_atan2.pos_neg_table = tl_pos_neg_table;
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

  if (m_no_negative) {
    m_p_mask.ifmap = tl_angle;
    m_p_mask.ofmap = tl_buf;
    m_p_mask.buf = tl_buf2;
    m_p_mask.pos_neg_table = tl_pos_neg_table;
    m_p_mask.fmt = FMT_BF16;

    m_p_mac_mask.res_is_int8 = 0;
    m_p_mac_mask.lshift_bits = 0;
    m_p_mac_mask.rshift_bits = 0;
    m_p_mac_mask.relu_enable = 0;
    m_p_mac_mask.bf16_enable = 1;
    m_p_mac_mask.res_high = NULL;
    m_p_mac_mask.res_low = tl_angle;
    m_p_mac_mask.a = tl_buf;
    m_p_mac_mask.b_is_const = 1;
    m_p_mac_mask.b_val = convert_fp32_bf16(360.f);
  }

  tl_in_idx->push_back(0);
  tl_in_idx->push_back(1);
  if (m_export_mag) {
    tl_out_idx->push_back(2);
  }
  if (m_export_ang) {
    tl_out_idx->push_back(3);
  }
#ifdef MAGnANG_DEBUG
  counting = 0;
#endif
  return BM_SUCCESS;
}

void IveTPUMagAndAng::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  if (m_export_mag) {
    bmk1880v2_tiu_bf16_element_wise_mul(bk_ctx, &m_p_mul);
    bmk1880v2_tiu_bf16_element_wise_mac(bk_ctx, &m_p_mac);
    bf16_emit_sqrt(bk_ctx, m_p_sqrt.a, m_p_sqrt.buf, m_p_sqrt.sqrt_table_answer,
                   m_p_sqrt.sqrt_table_answer_mantissa, m_p_sqrt.res);
  }
  if (m_export_ang) {
    // sqrt goes first cause atan2 changes y value
    bf16_atan2_emit(bk_ctx, m_p_atan2.b, m_p_atan2.a, m_p_atan2.buf1, m_p_atan2.buf2,
                    m_p_atan2.buf3, m_p_atan2.buf4, m_p_atan2.buf5, m_p_atan2.buf6, m_p_atan2.y0,
                    m_p_atan2.slope, m_p_atan2.invert, m_p_atan2.pos_neg_table,
                    m_p_atan2.reciprocal_table_answer, m_p_atan2.reciprocal_table_answer_mantissa,
                    m_p_atan2.sqrt_table_answer, m_p_atan2.sqrt_table_answer_mantissa,
                    m_p_atan2.idx_0_table, m_p_atan2.res, m_p_atan2.fmt);
#ifdef MAGnANG_DEBUG
    u16 *input1 = (u16 *)get_bf16_tensor_l2g(ctx, bk_ctx, m_p_atan2.a, FMT_BF16);
    u16 *input2 = (u16 *)get_bf16_tensor_l2g(ctx, bk_ctx, m_p_atan2.b, FMT_BF16);
    u16 *res_ang = (u16 *)get_bf16_tensor_l2g(ctx, bk_ctx, m_p_atan2.res, FMT_BF16);
    for (size_t i = 0; i < m_p_atan2.a->shape.h * m_p_atan2.a->shape.w; i++) {
      float b = convert_bf16_fp32(input2[i]);
      float a = convert_bf16_fp32(input1[i]);
      float resss = atan2(b, a);
      float err = (convert_bf16_fp32(res_ang[i]) - resss) / resss;
      if (err > 0.02) {
        printf("%lu ERRRRR b %f a %f cpu %f tpu %f\n", counting, b, a, resss,
               convert_bf16_fp32(res_ang[i]));
      }
    }
    delete[] input1;
    delete[] input2;
    delete[] res_ang;
#endif
    bmk1880v2_tiu_bf16_element_wise_mul(bk_ctx, &m_p_mul_const);

    if (m_no_negative) {
      // FIXME: Broken in TPU if no printf is inserted.
      // bf16_emit_mask_lt0(bk_ctx, m_p_mask.ifmap, m_p_mask.buf, m_p_mask.pos_neg_table,
      //                    m_p_mask.ofmap, m_p_mask.fmt);
      bf16_emit_neg_idx(bk_ctx, m_p_mask.ifmap, m_p_mask.buf, m_p_mask.pos_neg_table,
                        m_p_mask.ofmap, m_p_mask.fmt);
      bmk1880v2_tiu_bf16_element_wise_mac(bk_ctx, &m_p_mac_mask);
    }
  }
#ifdef MAGnANG_DEBUG
  counting++;
#endif
}