#pragma once
#include "core.hpp"

#include "bmkernel_non_atomic.h"

class IveTPUMadAndAng : public IveCore {
 public:
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;

 private:
  bmk1880v2_tiu_element_wise_mul_param_t m_p_mul;
  bmk1880v2_tiu_element_wise_mac_param_t m_p_mac;
  bmk1880v2_tiu_non_atomic_sqrt_param_t m_p_sqrt;
  bmk1880v2_tiu_non_atomic_atan2_param_t m_p_atan2;
  bmk1880v2_tiu_element_wise_mul_param_t m_p_mul_const;

  // TODO: Temporarily disable abs
  // bmk1880v2_tdma_tg2l_tensor_fill_constant_param_t m_p_fill;
  // bmk1880v2_tiu_element_wise_max_param_t m_p_max;
  // bmk1880v2_tiu_element_wise_min_param_t m_p_min;
  // bmk1880v2_tiu_element_wise_sub_param_t m_p_sub;
};