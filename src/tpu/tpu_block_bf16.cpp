#include "tpu/tpu_block.hpp"

#include <string.h>
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"

void IveTPUBlockBF16::setBinNum(const float bin_num) { m_bin_num = bin_num; }

void IveTPUBlockBF16::setCellSize(const int cell_size, const int channel) {
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

int IveTPUBlockBF16::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_slice_info.nums_of_tl = 3 * 2;
  m_kernel_info.nums_of_kernel = 1;

  return BM_SUCCESS;
}

int IveTPUBlockBF16::SliceSetup(SliceRes &slice_res, SliceRes *tg_in_res, SliceRes *tg_out_res) {
  *tg_in_res = slice_res;
  *tg_out_res = slice_res;
  tg_out_res->h.skip = tg_out_res->h.skip / m_kernel_info.size;
  tg_out_res->w.skip = tg_out_res->w.skip / m_kernel_info.size;
  tg_out_res->h.slice = tg_out_res->h.slice / m_kernel_info.size;
  tg_out_res->w.slice = tg_out_res->w.slice / m_kernel_info.size;
  return BM_SUCCESS;
}

int IveTPUBlockBF16::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
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
  bmk1880v2_tensor_lmem_shape_t tl_shape2;
  tl_shape2.n = tg_out_slices[0].n;
  tl_shape2.c = tg_out_slices[0].c;
  tl_shape2.h = tg_out_slices[0].h;
  tl_shape2.w = tg_out_slices[0].w;
  auto *tl_tmp = allocTLMem(bk_ctx, tl_shape2, FMT_BF16, 1);
  auto *tl_output = allocTLMem(bk_ctx, tl_shape2, FMT_BF16, 1);

  bmk1880v2_tensor_lmem_shape_t tl_block_shape;
  tl_block_shape.n = tg_out_slices[0].n;
  tl_block_shape.c = tg_out_slices[0].c;
  tl_block_shape.h = m_kernel_info.size;
  tl_block_shape.w = m_kernel_info.size;
  auto *block_kernel = allocTLMem(bk_ctx, tl_block_shape, FMT_BF16, 1);
  constantFillTL(ctx, bk_ctx, convert_fp32_bf16(1.f), block_kernel);
  float real_multiplier = 1.f / (m_kernel_info.size * m_kernel_info.size * m_bin_num);

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
  m_p_conv.ofmap = tl_tmp;
  m_p_conv.weight = block_kernel;

  m_p_mul.res_high = NULL;
  m_p_mul.res_low = tl_output;
  m_p_mul.a = tl_tmp;
  m_p_mul.b_is_const = 1;
  m_p_mul.b_val = convert_fp32_bf16(real_multiplier);
  m_p_mul.relu_enable = 0;
  m_p_mul.bf16_enable = 1;

  tl_in_idx->push_back(0);
  tl_out_idx->push_back(2);
  return BM_SUCCESS;
}

void IveTPUBlockBF16::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  bmk1880v2_tiu_bf16_depthwise_convolution(bk_ctx, &m_p_conv);
  bmk1880v2_tiu_bf16_element_wise_mul(bk_ctx, &m_p_mul);
}