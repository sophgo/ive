#include "table_manager.hpp"

TblMgr::TblMgr() {}
TblMgr::~TblMgr() {}

int TblMgr::init(bmctx_t *ctx, bmk1880v2_context_t *bk_ctx) {
  m_table_s_u8 = {1, 32, 16, 16};  // Kernel does not support U8 table info.
  // bf16_lut_tbl_bytesize(bk_ctx, &m_table_s_u8, FMT_U8);  // 16 * 16
  bf16_lut_tbl_bytesize(bk_ctx, &m_table_s, FMT_BF16);  // 32 * 8
  if (mp_atan_y0_degree == nullptr) {
    mp_atan_y0_degree = new CviImg(ctx, m_table_s.c, m_table_s.h, m_table_s.w, FMT_BF16);
    bf16_atan_fast_degree_y0((u16 *)mp_atan_y0_degree->GetVAddr(), &m_table_s);
    mp_atan_y0_degree->Flush(ctx);
  }
  if (mp_atan_y0 == nullptr) {
    mp_atan_y0 = new CviImg(ctx, m_table_s.c, m_table_s.h, m_table_s.w, FMT_BF16);
    bf16_atan_y0((u16 *)mp_atan_y0->GetVAddr(), &m_table_s);
    mp_atan_y0->Flush(ctx);
  }
  if (mp_atan_slope == nullptr) {
    mp_atan_slope = new CviImg(ctx, m_table_s.c, m_table_s.h, m_table_s.w, FMT_BF16);
    bf16_atan_slope((u16 *)mp_atan_slope->GetVAddr(), &m_table_s);
    mp_atan_slope->Flush(ctx);
  }
  if (mp_atan_invert == nullptr) {
    mp_atan_invert = new CviImg(ctx, m_table_s.c, m_table_s.h, m_table_s.w, FMT_BF16);
    bf16_atan_s_01((u16 *)mp_atan_invert->GetVAddr(), &m_table_s);
    mp_atan_invert->Flush(ctx);
  }

  if (mp_reciprocal_data == nullptr) {
    mp_reciprocal_data = new CviImg(ctx, m_table_s.c, m_table_s.h, m_table_s.w, FMT_BF16);
    bf16_gen_reciprocal((u16 *)mp_reciprocal_data->GetVAddr(), &m_table_s);
    mp_reciprocal_data->Flush(ctx);
  }
  if (mp_reciprocal_mantissa == nullptr) {
    mp_reciprocal_mantissa = new CviImg(ctx, m_table_s.c, m_table_s.h, m_table_s.w, FMT_BF16);
    bf16_gen_reciprocal_mantissa((u16 *)mp_reciprocal_mantissa->GetVAddr(), &m_table_s);
    mp_reciprocal_mantissa->Flush(ctx);
  }

  if (mp_sqrt_data == nullptr) {
    mp_sqrt_data = new CviImg(ctx, m_table_s.c, m_table_s.h, m_table_s.w, FMT_BF16);
    bf16_gen_sqrt((u16 *)mp_sqrt_data->GetVAddr(), &m_table_s);
    mp_sqrt_data->Flush(ctx);
  }
  if (mp_sqrt_mantissa == nullptr) {
    mp_sqrt_mantissa = new CviImg(ctx, m_table_s.c, m_table_s.h, m_table_s.w, FMT_BF16);
    bf16_gen_sqrt_mantissa((u16 *)mp_sqrt_mantissa->GetVAddr(), &m_table_s);
    mp_sqrt_mantissa->Flush(ctx);
  }

  if (mp_zero == nullptr) {
    mp_zero = new CviImg(ctx, m_table_s.c, m_table_s.h, m_table_s.w, FMT_BF16);
    bf16_gen_0_tbl((u16 *)mp_zero->GetVAddr(), &m_table_s);
    mp_zero->Flush(ctx);
  }
  if (mp_pos_neg == nullptr) {
    mp_pos_neg = new CviImg(ctx, m_table_s.c, m_table_s.h, m_table_s.w, FMT_BF16);
    bf16_atan_pos_neg((u16 *)mp_pos_neg->GetVAddr(), &m_table_s);
    mp_pos_neg->Flush(ctx);
  }
  return CVI_SUCCESS;
}

int TblMgr::free(bmctx_t *ctx) {
  if (mp_atan_y0_degree) {
    mp_atan_y0_degree->Free(ctx);
    delete mp_atan_y0_degree;
    mp_atan_y0_degree = nullptr;
  }
  if (mp_atan_y0) {
    mp_atan_y0->Free(ctx);
    delete mp_atan_y0;
    mp_atan_y0 = nullptr;
  }
  if (mp_atan_slope) {
    mp_atan_slope->Free(ctx);
    delete mp_atan_slope;
    mp_atan_slope = nullptr;
  }
  if (mp_atan_invert) {
    mp_atan_invert->Free(ctx);
    delete mp_atan_invert;
    mp_atan_invert = nullptr;
  }

  if (mp_reciprocal_data) {
    mp_reciprocal_data->Free(ctx);
    delete mp_reciprocal_data;
    mp_reciprocal_data = nullptr;
  }
  if (mp_reciprocal_mantissa) {
    mp_reciprocal_mantissa->Free(ctx);
    delete mp_reciprocal_mantissa;
    mp_reciprocal_mantissa = nullptr;
  }

  if (mp_sqrt_data) {
    mp_sqrt_data->Free(ctx);
    delete mp_sqrt_data;
    mp_sqrt_data = nullptr;
  }
  if (mp_sqrt_mantissa) {
    mp_sqrt_mantissa->Free(ctx);
    delete mp_sqrt_mantissa;
    mp_sqrt_mantissa = nullptr;
  }

  if (mp_pos_neg) {
    mp_pos_neg->Free(ctx);
    delete mp_pos_neg;
    mp_pos_neg = nullptr;
  }
  if (mp_zero) {
    mp_zero->Free(ctx);
    delete mp_zero;
    mp_zero = nullptr;
  }
  return CVI_SUCCESS;
}

const bmk1880v2_tensor_lmem_shape_t TblMgr::getTblTLShape(fmt_t fmt) {
  if (fmt == FMT_U8) {
    return m_table_s_u8;
  } else if (fmt == FMT_BF16) {
    return m_table_s;
  }
  std::cerr << "Unsupported fmt " << fmt << std::endl;
  return {0, 0, 0, 0};
}

const CviImg *TblMgr::atan(enum TBLATAN tblatan) {
  CviImg *img = nullptr;
  switch (tblatan) {
    case TBLATAN::TBLATAN_Y0: {
      img = mp_atan_y0;
    } break;
    case TBLATAN::TBLATAN_Y0_DEGREE: {
      img = mp_atan_y0_degree;
    } break;
    case TBLATAN::TBLATAN_SLOPE: {
      img = mp_atan_slope;
    } break;
    case TBLATAN::TBLATAN_INVERT: {
      img = mp_atan_invert;
    } break;
    case TBLATAN::TBLATAN_POSNEG: {
      img = mp_pos_neg;
    } break;
  }
  return img;
}

const CviImg *TblMgr::reciprocal(enum TBLRECIPROCAL tblrpc) {
  CviImg *img = nullptr;
  switch (tblrpc) {
    case TBLRECIPROCAL::TBLRECIPROCAL_DATA: {
      img = mp_reciprocal_data;
    } break;
    case TBLRECIPROCAL::TBLRECIPROCAL_MANTISSA: {
      img = mp_reciprocal_mantissa;
    } break;
  }
  return img;
}

const CviImg *TblMgr::sqrt(enum TBLSQRT tblsqrt) {
  CviImg *img = nullptr;
  switch (tblsqrt) {
    case TBLSQRT::TBLSQRT_DATA: {
      img = mp_sqrt_data;
    } break;
    case TBLSQRT::TBLSQRT_MANTISSA: {
      img = mp_sqrt_mantissa;
    } break;
  }
  return img;
}

const CviImg *TblMgr::mask(enum TBLMASK tblmask) {
  CviImg *img = nullptr;
  switch (tblmask) {
    case TBLMASK::TBLMASK_ZERO: {
      img = mp_zero;
    } break;
    case TBLMASK::TBLMASK_POSNEG: {
      img = mp_pos_neg;
    } break;
  }
  return img;
}