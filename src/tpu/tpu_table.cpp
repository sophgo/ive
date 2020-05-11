#include "tpu/tpu_table.hpp"
#include "utils.hpp"

#include <string.h>

void IveTPUTbl::setTable(bmctx_t *ctx, TblMgr *tblmgr, const u8 *tbl_data) {
  mp_tblmgr = tblmgr;
  auto &tl_shape_s = mp_tblmgr->getTblTLShape(FMT_U8);
  if (mp_table == nullptr) {
    mp_table = new CviImg(ctx, tl_shape_s.c, tl_shape_s.h, tl_shape_s.w, FMT_U8);
  }
  genTableU8(tl_shape_s, tbl_data, mp_table->GetVAddr());
  mp_table->Flush(ctx);
}

int IveTPUTbl::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_slice_info.io_fmt = FMT_U8;
  m_cmdbuf_subfix = "tbl";
  m_slice_info.ping_pong_size = 2;
  m_slice_info.ping_pong_share_tl = 0;
  m_slice_info.nums_of_tl = 1;
  m_slice_info.nums_of_table = 1;

  return CVI_SUCCESS;
}

int IveTPUTbl::runSetup(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx,
                        const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_in_slices,
                        const std::vector<bmk1880v2_tensor_tgmem_shape_t> &tg_out_slices,
                        std::vector<u32> *tl_in_idx, std::vector<u32> *tl_out_idx,
                        const bool enable_cext) {
  if (mp_table == nullptr) {
    std::cerr << "mp_table not set." << std::endl;
  }
  m_input.clear();
  bmk1880v2_tensor_lmem_shape_t tl_shape;
  tl_shape.n = tg_in_slices[0].n;
  tl_shape.c = tg_in_slices[0].c;
  tl_shape.h = tg_in_slices[0].h;
  tl_shape.w = tg_in_slices[0].w;
  for (size_t i = 0; i < m_slice_info.ping_pong_size; i++) {
    m_input.emplace_back(allocTLMem(bk_ctx, tl_shape, FMT_U8, 1));
  }

  bmk1880v2_tensor_lmem_shape_t tl_shape_s = mp_tblmgr->getTblTLShape(FMT_U8);
  auto tl_table = allocTLMem(bk_ctx, tl_shape_s, FMT_U8, 1, IVETLType::TABLE);
  cviImg2TL(ctx, bk_ctx, *mp_table, tl_table);

  m_p_tbl.table = tl_table;

  for (size_t pp = 0; pp < m_slice_info.ping_pong_size; pp++) {
    tl_in_idx->push_back(pp);
    tl_out_idx->push_back(pp);
  }
  return CVI_SUCCESS;
}

void IveTPUTbl::operation(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx, u32 ping_idx) {
  m_p_tbl.ifmap = m_input[ping_idx];
  m_p_tbl.ofmap = m_input[ping_idx];
  bmk1880v2_tiu_lookup_table(bk_ctx, &m_p_tbl);
}

int IveTPUTbl::freeChildTGMem(bmctx_t *ctx) {
  if (mp_table != nullptr) {
    mp_table->Free(ctx);
    delete mp_table;
    mp_table = nullptr;
  }
  return CVI_SUCCESS;
}