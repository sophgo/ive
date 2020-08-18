#pragma once
#include "core.hpp"
#include "table_manager.hpp"

class IveTPUTbl : public IveCore {
 public:
  void setTable(CVI_RT_HANDLE rt_handle, TblMgr *tblmgr, const uint8_t *tbl_data);
  virtual int init(CVI_RT_HANDLE rt_handle, cvk_context_t *cvk_ctx) override;

 protected:
  virtual int runSetup(CVI_RT_HANDLE rt_handle, cvk_context_t *cvk_ctx,
                       const std::vector<cvk_tg_shape_t> &tg_in_slices,
                       const std::vector<cvk_tg_shape_t> &tg_out_slices,
                       std::vector<uint32_t> *tl_in_idx, std::vector<uint32_t> *tl_out_idx,
                       const bool enable_cext) override;
  virtual void operation(CVI_RT_HANDLE rt_handle, cvk_context_t *cvk_ctx,
                         uint32_t ping_idx) override;
  virtual int postProcess(CVI_RT_HANDLE rt_handle) override;

 private:
  TblMgr *mp_tblmgr = nullptr;
  CviImg *mp_table = nullptr;
  std::vector<cvk_tl_t *> m_input;
  cvk_tiu_lookup_table_param_t m_p_tbl;
};
