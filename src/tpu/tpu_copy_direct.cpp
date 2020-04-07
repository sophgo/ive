#include <string.h>
#include "tpu/tpu_copy.hpp"

int IveTPUCopyDirect::run(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, std::vector<CviImg> &input,
                          std::vector<CviImg> *output) {
  if (input.size() != 1) {
    return CVI_FAILURE;
  }
  if (output->size() != 1) {
    return CVI_FAILURE;
  }
  bmk1880v2_tdma_tg2tg_tensor_copy_param_t copy_param;
  copy_param.src = &input[0].m_tg;
  copy_param.dst = &(*output)[0].m_tg;
  bmk1880v2_tdma_tg2tg_bf16_tensor_copy(bk_ctx, &copy_param);
  cviruntime_cvikernel_submit(*ctx);

  return CVI_SUCCESS;
}