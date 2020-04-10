#pragma once
#include "core.hpp"
#include "table_manager.hpp"

class IveTPUTbl : public IveCore {
 public:
  void setTable(bmctx_t *ctx, TblMgr *tblmgr, const u8 *tbl_data);
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
  TblMgr *mp_tblmgr = nullptr;
  CviImg *mp_table = nullptr;
  std::vector<bmk1880v2_tensor_lmem_t *> m_input;
  bmk1880v2_tiu_lookup_table_param_t m_p_tbl;
};
