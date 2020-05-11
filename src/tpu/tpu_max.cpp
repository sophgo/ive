#include "tpu/tpu_cmp.hpp"

#include <string.h>

void IveTPUMax::setKernelSize(u32 kz) {
  m_kernel_info.size = kz;
  m_kernel_info.pad[0] = 0;
  m_kernel_info.pad[1] = 0;
  m_kernel_info.pad[2] = 0;
  m_kernel_info.pad[3] = 0;
}

int IveTPUMax::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "max";
  m_slice_info.io_fmt = FMT_U8;
  m_slice_info.nums_of_tl = 2;
  m_kernel_info.nums_of_kernel = 0;
  return CVI_SUCCESS;
}

int IveTPUMax::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                        const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                        const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                        std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                        const bool enable_cext) {
  bmk1880v2_tensor_lmem_shape_t tl_shape, tl_shape_out;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  tl_shape_out.n = tg_out_slices[0].n;
  tl_shape_out.c = tg_out_slices[0].c;
  tl_shape_out.h = tg_out_slices[0].h;
  tl_shape_out.w = tg_out_slices[0].w;
  auto *tl_input = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);
  auto *tl_output = allocTLMem(bk_ctx, tl_shape_out, FMT_U8, 1);

  m_p_max.ifmap = tl_input;
  m_p_max.ofmap = tl_output;
  m_p_max.kh = m_kernel_info.size;
  m_p_max.kw = m_kernel_info.size;
  m_p_max.pad_top = 0;
  m_p_max.pad_bottom = 0;
  m_p_max.pad_left = 0;
  m_p_max.pad_right = 0;
  m_p_max.stride_w = 1;
  m_p_max.stride_h = 1;
  m_p_max.bf16_enable = 0;
  m_p_max.ins_fp = 0;  // Useless in max pooling but we still give it a value.

  tl_in_idx->push_back(0);
  tl_out_idx->push_back(1);
  return CVI_SUCCESS;
}

void IveTPUMax::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  bmk1880v2_tiu_max_pooling(bk_ctx, &m_p_max);
}