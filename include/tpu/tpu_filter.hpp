#pragma once
#include "core.hpp"

class IveTPUFilter : public IveCore {
 public:
  void setKernel(IveKernel &kernel);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) override;

 private:
  IveKernel *m_kernel = nullptr;
  bmk1880v2_tiu_depthwise_convolution_qdm_param_t m_p_conv;
};

class IveTPUFilterBF16 : public IveCore {
 public:
  void setKernel(const IveKernel &kernel);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) override;

 private:
  const IveKernel *m_kernel = nullptr;
  bmk1880v2_tiu_depthwise_convolution_param_t m_p_conv;
};