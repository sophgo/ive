#pragma once
#include "core.hpp"

class IveTPUMax : public IveCore {
 public:
  void setKernelSize(uint32_t kz);
  virtual int init(bmctx_t *ctx, cvk_context_t *cvk_ctx) override;

 protected:
  virtual int runSetup(bmctx_t *ctx, cvk_context_t *cvk_ctx,
                       const std::vector<cvk_tg_shape_t> &tg_in_slices,
                       const std::vector<cvk_tg_shape_t> &tg_out_slices,
                       std::vector<uint32_t> *tl_in_idx, std::vector<uint32_t> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, cvk_context_t *cvk_ctx, uint32_t ping_idx) override;

 private:
  cvk_tiu_max_pooling_param_t m_p_max;
};

class IveTPUMin : public IveCore {
 public:
  void setKernelSize(uint32_t kz);
  virtual int init(bmctx_t *ctx, cvk_context_t *cvk_ctx) override;

 protected:
  virtual int runSetup(bmctx_t *ctx, cvk_context_t *cvk_ctx,
                       const std::vector<cvk_tg_shape_t> &tg_in_slices,
                       const std::vector<cvk_tg_shape_t> &tg_out_slices,
                       std::vector<uint32_t> *tl_in_idx, std::vector<uint32_t> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, cvk_context_t *cvk_ctx, uint32_t ping_idx) override;

 private:
  cvk_tiu_mul_param_t m_p_mul;
  cvk_tiu_max_pooling_param_t m_p_max;
  cvk_tiu_mul_param_t m_p_mul_out;
};