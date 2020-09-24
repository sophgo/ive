#include "tpu/tpu_table.hpp"
#include "ive_log.hpp"
#include "utils.hpp"

#include <string.h>

void IveTPUTbl::setTable(CVI_RT_HANDLE rt_handle, TblMgr *tblmgr, const uint8_t *tbl_data) {
  mp_tblmgr = tblmgr;
  auto &tl_shape_s = mp_tblmgr->getTblTLShape(CVK_FMT_U8);
  if (mp_table == nullptr) {
    mp_table = new CviImg(rt_handle, tl_shape_s.c, tl_shape_s.h, tl_shape_s.w, CVK_FMT_U8);
  }
  genTableU8(tl_shape_s, tbl_data, mp_table->GetVAddr());
  mp_table->Flush(rt_handle);
}

int IveTPUTbl::init(CVI_RT_HANDLE rt_handle, cvk_context_t *cvk_ctx) {
  m_slice_info.io_fmt = CVK_FMT_U8;
  m_cmdbuf_subfix = "tbl";
  m_slice_info.ping_pong_size = 2;
  m_slice_info.ping_pong_share_tl = 0;
  m_slice_info.nums_of_tl = 1;
  m_slice_info.nums_of_table = 1;

  return CVI_SUCCESS;
}

int IveTPUTbl::runSetup(CVI_RT_HANDLE rt_handle, cvk_context_t *cvk_ctx,
                        const std::vector<cvk_tg_shape_t> &tg_in_slices,
                        const std::vector<cvk_tg_shape_t> &tg_out_slices,
                        std::vector<uint32_t> *tl_in_idx, std::vector<uint32_t> *tl_out_idx,
                        const bool enable_cext) {
  if (mp_table == nullptr) {
    LOGE("mp_table not set.\n");
  }
  m_input.clear();
  cvk_tl_shape_t tl_shape;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  for (size_t i = 0; i < m_slice_info.ping_pong_size; i++) {
    m_input.emplace_back(allocTLMem(cvk_ctx, tl_shape, CVK_FMT_U8, 1));
  }

  cvk_tl_shape_t tl_shape_s = mp_tblmgr->getTblTLShape(CVK_FMT_U8);
  auto tl_table = allocTLMem(cvk_ctx, tl_shape_s, CVK_FMT_U8, 1, IVETLType::TABLE);
  cviImg2TL(rt_handle, cvk_ctx, *mp_table, tl_table);

  m_p_tbl.table = tl_table;

  for (size_t pp = 0; pp < m_slice_info.ping_pong_size; pp++) {
    tl_in_idx->push_back(pp);
    tl_out_idx->push_back(pp);
  }
  return CVI_SUCCESS;
}

void IveTPUTbl::operation(CVI_RT_HANDLE rt_handle, cvk_context_t *cvk_ctx, uint32_t ping_idx) {
  m_p_tbl.ifmap = m_input[ping_idx];
  m_p_tbl.ofmap = m_input[ping_idx];
  cvk_ctx->ops->tiu_lookup_table(cvk_ctx, &m_p_tbl);
}

int IveTPUTbl::postProcess(CVI_RT_HANDLE rt_handle) {
  if (mp_table != nullptr) {
    mp_table->Free(rt_handle);
    delete mp_table;
    mp_table = nullptr;
  }
  return CVI_SUCCESS;
}