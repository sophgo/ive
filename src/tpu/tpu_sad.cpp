#include "tpu/tpu_sad.hpp"

#include <string.h>
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"

void IveTPUSAD::outputThresholdOnly(bool value) { m_output_thresh_only = value; }
void IveTPUSAD::doThreshold(bool value) { m_do_threshold = value; }

void IveTPUSAD::setThreshold(const u16 threshold, const u8 min_val, const u8 max_val) {
  m_threshold = threshold;
  m_min_value = min_val;
  m_max_value = max_val;
}

void IveTPUSAD::setCellSize(const int cell_size, const int channel) {
  m_kernel_info.size = cell_size;
  m_kernel_info.default_stride_x = cell_size;
  m_kernel_info.default_stride_y = cell_size;
  int pad = 0;
  m_kernel_info.pad[0] = pad;
  m_kernel_info.pad[1] = pad;
  m_kernel_info.pad[2] = pad;
  m_kernel_info.pad[3] = pad;
  m_channel = channel;
}

int IveTPUSAD::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  if (m_do_threshold) {
    m_slice_info.nums_of_tl = 5 * 2;
    m_slice_info.nums_of_table = 1 * 2;
  } else {
    m_slice_info.nums_of_tl = 4 * 2;
    m_slice_info.nums_of_table = 0;
  }
  m_kernel_info.nums_of_kernel = 1;

  return BM_SUCCESS;
}

int IveTPUSAD::SliceSetup(SliceRes &slice_res, SliceRes *tg_in_res, SliceRes *tg_out_res) {
  *tg_in_res = slice_res;
  *tg_out_res = slice_res;
  tg_out_res->h.skip = tg_out_res->h.skip / m_kernel_info.size;
  tg_out_res->w.skip = tg_out_res->w.skip / m_kernel_info.size;
  tg_out_res->h.slice = tg_out_res->h.slice / m_kernel_info.size;
  tg_out_res->w.slice = tg_out_res->w.slice / m_kernel_info.size;
  return BM_SUCCESS;
}

int IveTPUSAD::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                        const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                        const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                        std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) {
  if (m_channel != tg_in_slices[0].c) {
    std::cerr << "Channel changed, slicing result may not be suitable." << std::endl;
  }
  bmk1880v2_tensor_lmem_shape_t tl_shape;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  auto *tl_input = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  auto *tl_input2 = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);
  bmk1880v2_tensor_lmem_shape_t tl_shape2;
  tl_shape2.n = tg_out_slices[0].n;
  tl_shape2.c = tg_out_slices[0].c;
  tl_shape2.h = tg_out_slices[0].h;
  tl_shape2.w = tg_out_slices[0].w;
  auto *tl_output = allocTLMem(bk_ctx, tl_shape2, FMT_BF16, 1);
  auto *tl_output_thresh = allocTLMem(bk_ctx, tl_shape2, FMT_BF16, 1);

  auto *tl_abs_tmp = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);

  bmk1880v2_tensor_lmem_shape_t tl_block_shape;
  tl_block_shape.n = tg_out_slices[0].n;
  tl_block_shape.c = tg_out_slices[0].c;
  tl_block_shape.h = m_kernel_info.size;
  tl_block_shape.w = m_kernel_info.size;
  auto *block_kernel = allocTLMem(bk_ctx, tl_block_shape, FMT_BF16, 1);
  constantFillTL(ctx, bk_ctx, convert_fp32_bf16(1.f), block_kernel);

  m_p_min.a = tl_input;
  m_p_min.b = tl_input2;
  m_p_min.b_is_const = 0;
  m_p_min.bf16_enable = 1;
  m_p_min.min = tl_abs_tmp;

  m_p_max.a = tl_input;
  m_p_max.b = tl_input2;
  m_p_max.b_is_const = 0;
  m_p_max.bf16_enable = 1;
  m_p_max.max = tl_input;

  m_p_sub.a_high = NULL;
  m_p_sub.a_low = tl_input;
  m_p_sub.b_high = NULL;
  m_p_sub.b_low = tl_abs_tmp;
  m_p_sub.bf16_enable = 1;
  m_p_sub.res_high = NULL;
  m_p_sub.res_low = tl_input;
  m_p_sub.rshift_bits = 0;

  m_p_conv.pad_top = m_kernel_info.pad[2];
  m_p_conv.pad_bottom = m_kernel_info.pad[3];
  m_p_conv.pad_left = m_kernel_info.pad[0];
  m_p_conv.pad_right = m_kernel_info.pad[1];
  m_p_conv.stride_w = m_kernel_info.size;
  m_p_conv.stride_h = m_kernel_info.size;
  m_p_conv.relu_enable = 0;
  m_p_conv.ins_h = 0;
  m_p_conv.ins_w = 0;
  m_p_conv.ins_last_h = 0;
  m_p_conv.ins_last_w = 0;
  m_p_conv.dilation_h = 1;
  m_p_conv.dilation_w = 1;
  m_p_conv.bias = NULL;
  m_p_conv.rshift_bits = 0;
  m_p_conv.bf16_enable = 1;
  m_p_conv.ifmap = tl_input;
  m_p_conv.ofmap = tl_output;
  m_p_conv.weight = block_kernel;

  tl_in_idx->push_back(0);
  tl_in_idx->push_back(1);
  if (!m_output_thresh_only) {
    tl_out_idx->push_back(2);
  }

  if (m_do_threshold) {
    bmk1880v2_tensor_lmem_shape_t tl_table_s;
    bf16_lut_tbl_bytesize(bk_ctx, &tl_table_s, FMT_BF16);  // 32 * 8
    auto *tl_pos_neg_table = allocTLMem(bk_ctx, tl_table_s, FMT_BF16, 1);
    {
      CviImg table_data_atan_pos_neg(ctx, tl_table_s.c, tl_table_s.h, tl_table_s.w, FMT_BF16);
      genTableBF16((u16 *)table_data_atan_pos_neg.GetVAddr(), &tl_table_s, (float)m_min_value,
                   (float)m_max_value);
      bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
      p.src = &table_data_atan_pos_neg.m_tg;
      p.dst = tl_pos_neg_table;
      bmk1880v2_tdma_g2l_bf16_tensor_copy(bk_ctx, &p);
      bmruntime_bmkernel_submit(*ctx);
      table_data_atan_pos_neg.Free(ctx);
    }

    m_p_add_thresh.a_high = NULL;
    m_p_add_thresh.a_low = tl_output;
    m_p_add_thresh.b_is_const = 1;
    m_p_add_thresh.b_val = convert_fp32_bf16((-1 * m_threshold));
    m_p_add_thresh.bf16_enable = 1;
    m_p_add_thresh.res_high = NULL;
    m_p_add_thresh.res_low = tl_output_thresh;
    m_p_add_thresh.rshift_bits = 0;
    m_p_mask.ifmap = tl_output_thresh;
    m_p_mask.ofmap = tl_output_thresh;
    m_p_mask.pos_neg_table = tl_pos_neg_table;
    m_p_mask.fmt = FMT_BF16;

    tl_out_idx->push_back(3);
  }
  return BM_SUCCESS;
}

void IveTPUSAD::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  bmk1880v2_tiu_bf16_element_wise_min(bk_ctx, &m_p_min);
  bmk1880v2_tiu_bf16_element_wise_max(bk_ctx, &m_p_max);
  bmk1880v2_tiu_bf16_element_wise_sub(bk_ctx, &m_p_sub);
  bmk1880v2_tiu_bf16_depthwise_convolution(bk_ctx, &m_p_conv);
  if (m_do_threshold) {
    bmk1880v2_tiu_bf16_element_wise_add(bk_ctx, &m_p_add_thresh);
    bmk1880v2_tdma_l2l_tensor_copy_param_t p10;
    p10.dst = m_p_mask.ofmap;
    p10.src = m_p_mask.ifmap;
    p10.mv_lut_idx = true;
    bmk1880v2_tdma_l2l_bf16_tensor_copy(bk_ctx, &p10);
    p10.mv_lut_idx = false;

    bmk1880v2_tiu_lookup_table_param_t p12;
    p12.ofmap = m_p_mask.ofmap;
    p12.ifmap = m_p_mask.ofmap;
    p12.table = m_p_mask.pos_neg_table;
    bmk1880v2_tiu_lookup_table(bk_ctx, &p12);
  }
}