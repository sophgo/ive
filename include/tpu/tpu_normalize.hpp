#pragma once
#include "core.hpp"

class IveTPUNormalize : public IveCore {
 public:
  void setMinMax(float min, float max);
  void setOutputFMT(fmt_t fmt);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) override;

 private:
  float m_min = 0, m_max = 0;
  fmt_t m_fmt = FMT_INVALID;
  std::vector<bmk1880v2_tensor_lmem_t *> m_input;
  bmk1880v2_tiu_element_wise_add_param_t m_p_add;
  bmk1880v2_tiu_element_wise_mul_param_t m_p_mul;
  bmk1880v2_tiu_element_wise_add_param_t m_p_add_offset;
};