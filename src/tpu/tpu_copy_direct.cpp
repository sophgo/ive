#include <string.h>
#include "tpu/tpu_copy.hpp"

int IveTPUCopyDirect::run(bmctx_t *ctx, cvk_context_t *cvk_ctx, std::vector<CviImg> &input,
                          std::vector<CviImg> *output) {
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
  std::string name = "g2g_copy";
  submitCmdbuf(ctx, cvk_ctx, name);

  return CVI_SUCCESS;
}