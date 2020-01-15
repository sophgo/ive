#pragma once
#include "core.hpp"

class IveTPUThreshold : public IveCore {
 public:
  void setThreshold(int threshold);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;

 private:
  int m_threshold = -1;
  bmk1880v2_tiu_element_wise_mac_param_t m_p_mac;
  bmk1880v2_tiu_element_wise_mul_param_t m_p_mul;
};

class IveTPUThresholdHighLow : public IveCore {
 public:
  void setThreshold(int threshold, int low, int high);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;

 private:
  int m_threshold = -1;
  int m_threshold_high = 255;
  int m_threshold_low = 0;
  bmk1880v2_tiu_element_wise_mac_param_t m_p_mac;
  bmk1880v2_tiu_element_wise_mul_param_t m_p_mul;
  bmk1880v2_tiu_element_wise_max_param_t m_p_max;
  bmk1880v2_tiu_element_wise_min_param_t m_p_min;
};

class IveTPUThresholdSlope : public IveCore {
 public:
  void setThreshold(int low, int high);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;

 private:
  int m_threshold_high = 255;
  int m_threshold_low = 0;
  bmk1880v2_tiu_element_wise_max_param_t m_p_max;
  bmk1880v2_tiu_element_wise_min_param_t m_p_min;
};