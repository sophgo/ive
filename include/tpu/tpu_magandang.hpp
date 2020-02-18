#pragma once
#include "core.hpp"

#include "bmkernel_non_atomic.h"

class IveTPUMagAndAng : public IveCore {
 public:
  void exportOption(bool mag_value, bool ang_value);
  void noNegative(bool value);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;

 private:
  bool m_export_mag = true;
  bmk1880v2_tiu_element_wise_mul_param_t m_p_mul;
  bmk1880v2_tiu_element_wise_mac_param_t m_p_mac;
  bmk1880v2_tiu_non_atomic_sqrt_param_t m_p_sqrt;
  bool m_export_ang = true;
  bmk1880v2_tiu_non_atomic_atan2_param_t m_p_atan2;
  bmk1880v2_tiu_element_wise_mul_param_t m_p_mul_const;

  bool m_no_negative = false;
  bmk1880v2_tiu_non_atomic_mask_param_t m_p_mask;
  bmk1880v2_tiu_element_wise_mac_param_t m_p_mac_mask;
};