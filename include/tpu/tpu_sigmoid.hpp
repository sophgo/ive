#pragma once
#include "core.hpp"

typedef struct bmk1880v2_tiu_non_atomic_sigmoid_param {
  float scale;
  bmk1880v2_tensor_lmem_t *ifmap;
  bmk1880v2_tensor_lmem_t *buf;
  bmk1880v2_tensor_lmem_t *table_answer;
  bmk1880v2_tensor_lmem_t *table_answer_slope;
  bmk1880v2_tensor_lmem_t *ofmap;
} bmk1880v2_tiu_non_atomic_sigmoid_param_t;

class IveTPUSigmoid : public IveCore {
 public:
  virtual int init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) override;

 protected:
  virtual int runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                       const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                       std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) override;
  virtual int freeChildTGMem(bmctx_t *ctx) override;

 private:
  CviImg *table = nullptr, *table_slope = nullptr;
  std::vector<bmk1880v2_tensor_lmem_t *> m_input;
  std::vector<bmk1880v2_tensor_lmem_t *> m_buf;
  std::vector<bmk1880v2_tensor_lmem_t *> m_output;
  bmk1880v2_tiu_non_atomic_sigmoid_param_t m_p_sig;
};