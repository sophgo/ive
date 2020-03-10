#pragma once
#include "core.hpp"

class IveTPUAnd : public IveCore {
 public:
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) override;

 private:
  std::vector<bmk1880v2_tensor_lmem_t *> m_input1;
  std::vector<bmk1880v2_tensor_lmem_t *> m_input2;
  bmk1880v2_tiu_element_wise_and_int8_param_t m_p_and;
};