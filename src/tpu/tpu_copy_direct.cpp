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
  copy_param.src = &input[0].m_tg;
  copy_param.dst = &(*output)[0].m_tg;
  cvk_ctx->ops->tdma_g2g_bf16_tensor_copy(cvk_ctx, &copy_param);
  CVI_RT_Submit(cvk_ctx);

  return CVI_SUCCESS;
}