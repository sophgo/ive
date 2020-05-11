#include "tpu/tpu_morph.hpp"
#include <string.h>

void IveTPUErode::setKernel(IveKernel &kernel) {
  m_kernel = &kernel;
  m_kernel_info.size = m_kernel->img.m_tg.shape.h;
  int pad = (m_kernel_info.size - 1) / 2;
  m_kernel_info.pad[0] = pad;
  m_kernel_info.pad[1] = pad;
  m_kernel_info.pad[2] = pad;
  m_kernel_info.pad[3] = pad;
}

int IveTPUErode::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_cmdbuf_subfix = "morph";
  m_slice_info.io_fmt = FMT_U8;
  m_slice_info.nums_of_tl = 3;
  m_kernel_info.nums_of_kernel = 1;
  return CVI_SUCCESS;
}

int IveTPUErode::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
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
  auto *tl_conv_res = allocTLMem(bk_ctx, tl_shape_out, FMT_U8, 1);
  // Kernel
  if (m_kernel == nullptr) {
    std::cerr << "Error! kernel not set." << std::endl;
  }
  bmk1880v2_tensor_lmem_shape_t tl_kernel_s = {1, tl_shape.c, m_kernel_info.size,
                                               m_kernel_info.size};
  bmk1880v2_tensor_lmem_shape_t packed_s = {1, tl_shape.c, 1, MULTIPLIER_ONLY_PACKED_DATA_SIZE};
  auto *tl_kernel = allocTLMem(bk_ctx, tl_kernel_s, FMT_U8, 1, IVETLType::KERNEL);
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
    tl_multiplier->stride = bmk1880v2_tensor_lmem_default_stride(bk_ctx, m_tl_vec[3]->shape, 0);
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
  m_p_conv.ofmap = tl_conv_res;
  m_p_conv.weight = tl_kernel;
  m_p_conv.chl_quan_param = tl_multiplier;

  mp_xor_ones = allocTLMem(bk_ctx, tl_shape, FMT_U8, 1);
  constantFillTL(ctx, bk_ctx, 255, mp_xor_ones);

  m_p_xor.b = mp_xor_ones;
  m_p_xor.layer_id = 0;

  tl_in_idx->push_back(0);
  tl_out_idx->push_back(1);
  return CVI_SUCCESS;
}

void IveTPUErode::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  m_p_xor.a = m_tl_vec[0];
  m_p_xor.res = m_tl_vec[0];
  mp_xor_ones->shape = m_tl_vec[0]->shape;
  mp_xor_ones->stride = m_tl_vec[0]->stride;
  bmk1880v2_tiu_element_wise_xor_int8(bk_ctx, &m_p_xor);
  bmk1880v2_tiu_depthwise_convolution_qdm(bk_ctx, &m_p_conv);
  m_p_xor.a = m_tl_vec[1];
  m_p_xor.res = m_tl_vec[1];
  mp_xor_ones->shape = m_tl_vec[1]->shape;
  mp_xor_ones->stride = m_tl_vec[1]->stride;
  bmk1880v2_tiu_element_wise_xor_int8(bk_ctx, &m_p_xor);
}

int IveTPUErode::freeChildTGMem(bmctx_t *ctx) {
  mp_multiplier->Free(ctx);
  delete mp_multiplier;
  mp_multiplier = nullptr;
  return CVI_SUCCESS;
}