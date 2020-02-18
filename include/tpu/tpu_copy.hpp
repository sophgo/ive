#pragma once
#include "core.hpp"

class IveTPUCopyDirect {
 public:
  static int run(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, std::vector<CviImg> &input,
                 std::vector<CviImg> *output);
};

class IveTPUCopyInterval : public IveCore {
 public:
  void setInvertal(u32 hori, u32 verti);
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;
  virtual int SliceSetup(SliceRes &slice_res, SliceRes *tg_in_res, SliceRes *tg_out_res) override;
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;

 private:
  u32 m_hori = 1;
  u32 m_verti = 1;
  bmk1880v2_tiu_element_wise_copy_param_t m_p_copy;
};