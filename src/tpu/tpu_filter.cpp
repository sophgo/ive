#include "tpu/tpu_filter.hpp"
#include <string.h>

void IveTPUFilter::setKernel(IveKernel &kernel) {
  m_kernel = &kernel;
  m_kernel_info.size = m_kernel->img.m_tg.shape.h;
  int pad = (m_kernel_info.size - 1) / 2;
  m_kernel_info.pad[0] = pad;
  m_kernel_info.pad[1] = pad;
  m_kernel_info.pad[2] = pad;
  m_kernel_info.pad[3] = pad;
}

int IveTPUFilter::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_slice_info.io_fmt = FMT_U8;
  m_slice_info.nums_of_tl = 2;
  m_kernel_info.nums_of_kernel = 1;
  return BM_SUCCESS;
}

int IveTPUFilter::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
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

  // Kernel
  if (m_kernel == nullptr) {
    std::cerr << "Error! kernel not set." << std::endl;
  }
  bmk1880v2_tensor_lmem_shape_t tl_kernel_s = {1, tl_shape.c, m_kernel_info.size,
                                               m_kernel_info.size};
  bmk1880v2_tensor_lmem_shape_t packed_s = {1, tl_shape.c, 1, MULTIPLIER_ONLY_PACKED_DATA_SIZE};
  auto *tl_kernel = allocTLMem(bk_ctx, tl_kernel_s, FMT_U8, 1, IVETLType::KERNEL);
  if (m_kernel->img.m_tg.shape.c < tl_shape.c) {
    std::cerr << "kernel size must larger than tl_shape.c" << std::endl;
    return BM_ERR_FAILURE;
  }
  int tmp_c = m_kernel->img.m_tg.shape.c;
  m_kernel->img.m_tg.shape.c = tl_shape.c;
  cviImgFlush2TL(ctx, bk_ctx, m_kernel->img, tl_kernel);
  m_kernel->img.m_tg.shape.c = tmp_c;

  auto *tl_multiplier = allocTLMem(bk_ctx, packed_s, FMT_U8, 1);
  {
    mp_multiplier = new CviImg(ctx, tl_shape.c, 1, MULTIPLIER_ONLY_PACKED_DATA_SIZE, FMT_U8);
    getPackedMultiplierArrayBuffer(tl_shape.c, m_kernel->multiplier.base,
                                   m_kernel->multiplier.shift, mp_multiplier->GetVAddr());
    cviImgFlush2TL(ctx, bk_ctx, *mp_multiplier, tl_multiplier);
    tl_multiplier->shape = {1, tl_shape.c, 1, 1};
    tl_multiplier->stride = bmk1880v2_tensor_lmem_default_stride(bk_ctx, tl_multiplier->shape, 0);
  }

  if (enable_cext) {
    m_p_conv.pad_top = 0;
    m_p_conv.pad_bottom = 0;
  } else {
    m_p_conv.pad_top = m_kernel_info.pad[2];
    m_p_conv.pad_bottom = m_kernel_info.pad[3];
  }
  m_p_conv.pad_left = m_kernel_info.pad[0];
  m_p_conv.pad_right = m_kernel_info.pad[1];
  m_p_conv.stride_w = 1;
  m_p_conv.stride_h = 1;
  m_p_conv.relu_enable = 1;
  m_p_conv.ins_h = 0;
  m_p_conv.ins_w = 0;
  m_p_conv.ins_last_h = 0;
  m_p_conv.ins_last_w = 0;
  m_p_conv.dilation_h = 1;
  m_p_conv.dilation_w = 1;
  m_p_conv.has_bias = 0;

  m_p_conv.ifmap = tl_input;
  m_p_conv.ofmap = tl_output;
  m_p_conv.weight = tl_kernel;
  m_p_conv.chl_quan_param = tl_multiplier;

  tl_in_idx->push_back(0);
  tl_out_idx->push_back(1);
  return BM_SUCCESS;
}

void IveTPUFilter::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  bmk1880v2_tiu_depthwise_convolution_qdm(bk_ctx, &m_p_conv);
}

int IveTPUFilter::freeChildTGMem(bmctx_t *ctx) {
  mp_multiplier->Free(ctx);
  delete mp_multiplier;
  mp_multiplier = nullptr;
  return BM_SUCCESS;
}

void IveTPUFilterBF16::setKernel(const IveKernel &kernel) {
  m_kernel = &kernel;
  m_kernel_info.size = m_kernel->img.m_tg.shape.h;
  int pad = (m_kernel_info.size - 1) / 2;
  m_kernel_info.pad[0] = pad;
  m_kernel_info.pad[1] = pad;
  m_kernel_info.pad[2] = pad;
  m_kernel_info.pad[3] = pad;
}

int IveTPUFilterBF16::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "filter";
  m_slice_info.nums_of_tl = 2 * 2;
  m_kernel_info.nums_of_kernel = 1;
  return BM_SUCCESS;
}

int IveTPUFilterBF16::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
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
  auto *tl_output = allocTLMem(bk_ctx, tl_shape, FMT_BF16, 1);

  // Kernel
  if (m_kernel == nullptr) {
    std::cerr << "Error! kernel not set." << std::endl;
  }
  bmk1880v2_tensor_lmem_shape_t tl_kernel_s = {1, tl_shape.c, m_kernel_info.size,
                                               m_kernel_info.size};
  bmk1880v2_tensor_lmem_shape_t packed_s = {1, tl_shape.c, 1, MULTIPLIER_ONLY_PACKED_DATA_SIZE};
  auto *tl_kernel = allocTLMem(bk_ctx, tl_kernel_s, FMT_BF16, 1, IVETLType::KERNEL);
  {
    bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
    p.src = &m_kernel->img.m_tg;
    p.dst = tl_kernel;
    bmk1880v2_tdma_g2l_tensor_copy(bk_ctx, &p);
  }

  m_p_conv.pad_top = m_kernel_info.pad[2];
  m_p_conv.pad_bottom = m_kernel_info.pad[3];
  m_p_conv.pad_left = m_kernel_info.pad[0];
  m_p_conv.pad_right = m_kernel_info.pad[1];
  m_p_conv.stride_w = 1;
  m_p_conv.stride_h = 1;
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
  m_p_conv.weight = tl_kernel;

  tl_in_idx->push_back(0);
  tl_out_idx->push_back(1);
  return BM_SUCCESS;
}

void IveTPUFilterBF16::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  bmk1880v2_tiu_depthwise_convolution(bk_ctx, &m_p_conv);
}