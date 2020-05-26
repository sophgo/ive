#pragma once
#include "core.hpp"

class IveTPUFilter : public IveCore {
 public:
  void setKernel(IveKernel &kernel);
  virtual int init(bmctx_t *ctx, cvk_context_t *cvk_ctx) override;

 protected:
  virtual int runSetup(bmctx_t *ctx, cvk_context_t *cvk_ctx,
                       const std::vector<cvk_tg_shape_t> &tg_in_slices,
                       const std::vector<cvk_tg_shape_t> &tg_out_slices,
                       std::vector<uint32_t> *tl_in_idx, std::vector<uint32_t> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, cvk_context_t *cvk_ctx, uint32_t ping_idx) override;
  virtual int postProcess(bmctx_t *ctx) override;

 private:
  IveKernel *m_kernel = nullptr;
  CviImg *mp_multiplier = nullptr;
  cvk_tiu_depthwise_convolution_param_t m_p_conv;
};

class IveTPUFilterBF16 : public IveCore {
 public:
  void setKernel(const IveKernel &kernel);
  virtual int init(bmctx_t *ctx, cvk_context_t *cvk_ctx) override;
  virtual int runSetup(bmctx_t *ctx, cvk_context_t *cvk_ctx,
                       const std::vector<cvk_tg_shape_t> &tg_in_slices,
                       const std::vector<cvk_tg_shape_t> &tg_out_slices,
                       std::vector<uint32_t> *tl_in_idx, std::vector<uint32_t> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, cvk_context_t *cvk_ctx, uint32_t ping_idx) override;

 private:
  const IveKernel *m_kernel = nullptr;
  cvk_tiu_depthwise_pt_convolution_param_t m_p_conv;
};