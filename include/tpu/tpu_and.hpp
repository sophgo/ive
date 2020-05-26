#pragma once
#include "core.hpp"

class IveTPUAnd : public IveCore {
 public:
  virtual int init(bmctx_t *ctx, cvk_context_t *cvk_ctx) override;

 protected:
  virtual int runSetup(bmctx_t *ctx, cvk_context_t *cvk_ctx,
                       const std::vector<cvk_tg_shape_t> &tg_in_slices,
                       const std::vector<cvk_tg_shape_t> &tg_out_slices,
                       std::vector<uint32_t> *tl_in_idx, std::vector<uint32_t> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, cvk_context_t *cvk_ctx, uint32_t ping_idx) override;

 private:
  std::vector<cvk_tl_t *> m_input1;
  std::vector<cvk_tl_t *> m_input2;
  cvk_tiu_and_int8_param_t m_p_and;
};