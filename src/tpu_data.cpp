#include "tpu_data.hpp"

CviImg::CviImg() {}
CviImg::CviImg(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt) {
  Init(ctx, img_c, img_h, img_w, fmt);
}

CviImg::CviImg(bmctx_t *ctx, const CviImg &img, u32 x1, u32 y1, u32 x2, u32 y2) {
  this->m_tg = img.m_tg;
  this->m_bmmem = img.m_bmmem;
  this->m_size = img.m_size;
  u32 x2_new = x2 > img.m_tg.shape.w ? img.m_tg.shape.w : x2;
  u32 y2_new = y2 > img.m_tg.shape.h ? img.m_tg.shape.h : y2;
  u32 new_width = x2_new - x1;
  u32 new_height = y2_new - y1;

  // Update subimage shape
  this->m_tg.shape.h = new_height;
  this->m_tg.shape.w = new_width;
  u32 start_offset = y1 * img.m_tg.stride.h + x1 * getFmtSize(img.m_tg.fmt);
  this->m_tg.start_address = img.m_tg.start_address + start_offset;
  this->m_paddr = img.m_paddr + start_offset;
  this->m_vaddr = img.m_vaddr + start_offset;
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
  return AllocateDevice(ctx);
}

const bool CviImg::IsInit() { return this->m_bmmem == NULL ? false : true; }

uint8_t *CviImg::GetVAddr() { return m_vaddr; }

uint64_t CviImg::GetPAddr() const { return m_paddr; }

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
      case FMT_U16:
      case FMT_I16: {
        bms = BM_TENSOR_INT16((int)m_tg.shape.n, (int)m_tg.shape.c, (int)m_tg.shape.h,
                              (int)m_tg.shape.w);
      } break;
      case FMT_BF16: {
        bms = BM_TENSOR_BF16((int)m_tg.shape.n, (int)m_tg.shape.c, (int)m_tg.shape.h,
                             (int)m_tg.shape.w);
      } break;
      case FMT_F32: {
        bms = BM_TENSOR_FP32((int)m_tg.shape.n, (int)m_tg.shape.c, (int)m_tg.shape.h,
                             (int)m_tg.shape.w);
      } break;
      default: { std::cerr << "Unsupported bmshape_t type" << std::endl; } break;
    }

    this->m_bmmem = bmmem_device_alloc(*ctx, &bms);
    this->m_tg.base_reg_index = 0;
    this->m_tg.start_address = bmmem_device_addr(*ctx, this->m_bmmem);
    this->m_tg.stride = bmk1880v2_bf16_tensor_tgmem_default_stride(m_tg.shape, m_tg.fmt);
  }
  m_vaddr = (uint8_t *)bmmem_device_v_addr(*ctx, this->m_bmmem);
  m_paddr = bmmem_device_addr(*ctx, this->m_bmmem);
  return ret;
}

int CviImg::Free(bmctx_t *ctx) {
  bmmem_device_free(*ctx, this->m_bmmem);
  return 0;
}