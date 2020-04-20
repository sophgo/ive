#pragma once
#include "core.hpp"
#include "table_manager.hpp"

#include "bmkernel_non_atomic.h"

class IveTPUSobel : public IveCore {
 public:
  void setTblMgr(TblMgr *tblmgr);
  void setKernel(IveKernel &kernel_x, IveKernel &kernel_y);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) override;

 private:
  TblMgr *mp_tblmgr = nullptr;
  IveKernel *m_kernel_x = nullptr;
  IveKernel *m_kernel_y = nullptr;
  bmk1880v2_tiu_depthwise_convolution_param_t m_p_conv;
  bmk1880v2_tiu_element_wise_mul_param_t m_p_mul;
  bmk1880v2_tiu_element_wise_mac_param_t m_p_mac;
  bmk1880v2_tiu_non_atomic_sqrt_param_t m_p_sqrt;
};

class IveTPUSobelGradOnly : public IveCore {
 public:
  void setKernel(IveKernel &kernel_x, IveKernel &kernel_y);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;

 protected:
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) override;

 private:
  IveKernel *m_kernel_x = nullptr;
  IveKernel *m_kernel_y = nullptr;
  bmk1880v2_tiu_depthwise_convolution_param_t m_p_conv;
};