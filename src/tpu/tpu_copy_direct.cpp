#include <string.h>
#include "tpu/tpu_copy.hpp"

int IveTPUCopyDirect::run(CVI_RT_HANDLE rt_handle, cvk_context_t *cvk_ctx,
                          std::vector<CviImg> &input, std::vector<CviImg> *output) {
  if (input.size() != 1) {
    return CVI_FAILURE;
  }
  if (output->size() != 1) {
    return CVI_FAILURE;
  }
  cvk_tdma_g2g_tensor_copy_param_t copy_param;
  if (input[0].m_tg.shape.h > 4096 || input[0].m_tg.shape.w > 4096) {
    cvk_tg_t in_tg = input[0].m_tg;
    cvk_tg_t out_tg = (*output)[0].m_tg;
    copy_param.src = &in_tg;
    copy_param.dst = &out_tg;
#ifdef WORKAROUND_SCALAR_4096_ALIGN_BUG
    uint32_t value = 4096;
#else
    uint32_t value = input[0].GetImgStrides()[0] > 4096 ? 4096 : input[0].GetImgStrides()[0];
#endif
    uint32_t total_size = input[0].m_tg.shape.n * input[0].m_tg.shape.c * input[0].m_tg.shape.h *
                          input[0].m_tg.shape.w;
    uint32_t h_turns = total_size / value;
    uint32_t c_turns = h_turns / 4096;
    if (c_turns > 0) {
      in_tg.shape.w = value;
      in_tg.shape.h = 4096;
      in_tg.shape.c = c_turns;
      in_tg.shape.n = 1;
      in_tg.stride = cvk_ctx->ops->tg_default_stride(cvk_ctx, in_tg.shape, in_tg.fmt);
      out_tg.shape = in_tg.shape;
      out_tg.stride = in_tg.stride;
      cvk_ctx->ops->tdma_g2g_bf16_tensor_copy(cvk_ctx, &copy_param);
      in_tg.start_address += in_tg.stride.n;
      out_tg.start_address += in_tg.stride.n;
    }
    uint32_t h_left = h_turns - c_turns * 4096;
    if (h_left > 0) {
      in_tg.shape.w = value;
      in_tg.shape.h = h_left;
      in_tg.shape.c = 1;
      in_tg.shape.n = 1;
      in_tg.stride = cvk_ctx->ops->tg_default_stride(cvk_ctx, in_tg.shape, in_tg.fmt);
      out_tg.shape = in_tg.shape;
      out_tg.stride = in_tg.stride;
      cvk_ctx->ops->tdma_g2g_bf16_tensor_copy(cvk_ctx, &copy_param);
      in_tg.start_address += in_tg.stride.n;
      out_tg.start_address += in_tg.stride.n;
    }
    uint32_t w_left = total_size - h_turns * value;
    if (w_left > 0) {
      in_tg.shape.w = w_left;
      in_tg.shape.h = 1;
      in_tg.shape.c = 1;
      in_tg.shape.n = 1;
      in_tg.stride = cvk_ctx->ops->tg_default_stride(cvk_ctx, in_tg.shape, in_tg.fmt);
      out_tg.shape = in_tg.shape;
      out_tg.stride = in_tg.stride;
      cvk_ctx->ops->tdma_g2g_bf16_tensor_copy(cvk_ctx, &copy_param);
    }
  } else {
    copy_param.src = &input[0].m_tg;
    copy_param.dst = &(*output)[0].m_tg;
    cvk_ctx->ops->tdma_g2g_bf16_tensor_copy(cvk_ctx, &copy_param);
  }

  CVI_RT_Submit(cvk_ctx);

  return CVI_SUCCESS;
}