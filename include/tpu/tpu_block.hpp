#pragma once
#include "core.hpp"

class IveTPUBlock : public IveCore {
 public:
  void setBinNum(const float bin_num);
  void setCellSize(const int cell_size, const int channel = 3);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;

 protected:
  virtual int sliceSetup(SliceRes &slice_res, SliceRes *tg_in_res, SliceRes *tg_out_res) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) override;
  virtual int freeChildTGMem(bmctx_t *ctx) override;

 private:
  float m_bin_num = 1;
  u32 m_channel = 3;
  CviImg *mp_multiplier = nullptr;
  bmk1880v2_tiu_depthwise_convolution_qdm_param_t m_p_conv;
};

class IveTPUBlockBF16 : public IveCore {
 public:
  void setBinNum(const float bin_num);
  void setCellSize(const int cell_size, const int channel = 3);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;

 protected:
  virtual int sliceSetup(SliceRes &slice_res, SliceRes *tg_in_res, SliceRes *tg_out_res) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) override;

 private:
  float m_bin_num = 1;
  u32 m_channel = 3;
  bmk1880v2_tiu_depthwise_convolution_param_t m_p_conv;
  bmk1880v2_tiu_element_wise_mul_param_t m_p_mul;
};