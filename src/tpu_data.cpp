#include "tpu_data.hpp"

CviImg::CviImg() {}
CviImg::CviImg(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt) {
  Init(ctx, img_c, img_h, img_w, fmt);
}

CviImg::CviImg(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt, bmmem_device_t bmmem) {
  this->m_bmmem = bmmem;
  Init(ctx, img_c, img_h, img_w, fmt);
}

CviImg::CviImg(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt, u8 *data) {
  Init(ctx, img_c, img_h, img_w, fmt);
  memcpy(this->m_vaddr, data, this->m_size);
}

int CviImg::Init(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt) {
  this->m_tg.shape = {1, img_c, img_h, img_w};
  this->m_tg.fmt = fmt;
  this->m_size = this->m_tg.shape.n * this->m_tg.shape.c * this->m_tg.shape.h * this->m_tg.shape.w *
                 getFmtSize(m_tg.fmt);
  AllocateDevice(ctx);
  return 0;
}

const bool CviImg::IsInit() { return this->m_bmmem == NULL ? false : true; }

uint8_t *CviImg::GetVAddr() { return m_vaddr; }

uint64_t CviImg::GetPAddr() { return m_paddr; }

const u64 CviImg::GetImgSize() { return m_size; }

int CviImg::AllocateDevice(bmctx_t *ctx) {
  int ret = 1;
  if (this->m_bmmem == NULL) {
    ret = 0;
    bmshape_t bms;
    switch (m_tg.fmt) {
      case FMT_U8:
      case FMT_I8: {
        bms = BM_TENSOR_INT8((int)m_tg.shape.n, (int)m_tg.shape.c, (int)m_tg.shape.h,
                             (int)m_tg.shape.w);
      } break;
      case FMT_BF16: {
        bms = BM_TENSOR_BF16((int)m_tg.shape.n, (int)m_tg.shape.c, (int)m_tg.shape.h,
                             (int)m_tg.shape.w);
      } break;
      default: {
        std::cerr << "Unsupported bmshape_t type" << std::endl;
      } break;
    }

    this->m_bmmem = bmmem_device_alloc(*ctx, &bms);
    this->m_tg.base_reg_index = 0;
    this->m_tg.start_address = bmmem_device_addr(*ctx, this->m_bmmem);
    this->m_tg.stride = bmk1880v2_tensor_tgmem_default_stride(m_tg.shape);
  }
  m_vaddr = bmmem_device_v_addr(*ctx, this->m_bmmem);
  m_paddr = bmmem_device_addr(*ctx, this->m_bmmem);
  return ret;
}

int CviImg::Free(bmctx_t *ctx) {
  bmmem_device_free(*ctx, this->m_bmmem);
  return 0;
}