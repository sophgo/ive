#include "tpu_data.hpp"

CviImg::CviImg() {}
CviImg::CviImg(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt) {
  Init(ctx, img_c, img_h, img_w, fmt);
}

CviImg::CviImg(bmctx_t *ctx, const CviImg &img, u32 x1, u32 y1, u32 x2, u32 y2) {
  if (!this->m_is_stride_ceq) {
    std::cerr << "Error, sub-image does not support non-equal stride in different channels."
              << std::endl;
  }
  if (x1 > x2) {
    u32 tmp = x1;
    x1 = x2;
    x2 = tmp;
  }
  if (y1 > y2) {
    u32 tmp = y1;
    y1 = y2;
    y2 = tmp;
  }
  u32 x1_new = x1 > img.m_tg.shape.w ? img.m_tg.shape.w : x1;
  u32 y1_new = y1 > img.m_tg.shape.h ? img.m_tg.shape.h : y1;
  u32 x2_new = x2 > img.m_tg.shape.w ? img.m_tg.shape.w : x2;
  u32 y2_new = y2 > img.m_tg.shape.h ? img.m_tg.shape.h : y2;
  u32 new_width = x2_new - x1_new;
  u32 new_height = y2_new - y1_new;
  if (new_width == 0 || new_height == 0) {
    std::cerr << "Error width " << new_width << " or height " << new_height << " cannot be 0."
              << std::endl;
  }

  // Update subimage shape
  this->m_bmmem = img.m_bmmem;
  this->m_size = img.m_size;
  this->m_fmt = img.m_fmt;
  this->m_channel = img.m_channel;
  this->m_width = new_width;
  this->m_height = new_height;
  this->m_strides = img.m_strides;
  this->m_heights = img.m_heights;
  this->m_img_type = img.m_img_type;
  this->m_is_planar = img.m_is_planar;
  this->m_is_stride_ceq = img.m_is_stride_ceq;

  this->m_tg = img.m_tg;
  this->m_tg.shape.h = new_height;
  this->m_tg.shape.w = new_width;
  u32 start_offset = y1_new * img.m_tg.stride.h + x1_new * getFmtSize(img.m_tg.fmt);
  this->m_tg.start_address = img.m_tg.start_address + start_offset;
  this->m_paddr = img.m_paddr + start_offset;
  this->m_vaddr = img.m_vaddr + start_offset;
  this->m_is_sub_img = true;
}

CviImg::CviImg(bmctx_t *ctx, u32 img_h, u32 img_w, std::vector<u32> strides,
               std::vector<u32> heights, CVIIMGTYPE img_type, fmt_t fmt) {
  if (strides.size() == 0) {
    std::cerr << "No stride given." << std::endl;
    return;
  }
  if (strides.size() != heights.size()) {
    std::cerr << "Strides size and heights size must be the same." << std::endl;
    return;
  }
  this->m_fmt = fmt;
  this->m_channel = strides.size();
  this->m_width = img_w;
  this->m_height = img_h;
  this->m_strides = strides;
  this->m_heights = heights;
  this->m_img_type = img_type;
  this->m_is_planar = IsImgPlanar(this->m_img_type);
  this->m_size = strides[0] * heights[0];
  for (size_t i = 1; i < strides.size(); i++) {
    if (strides[i] != strides[0]) {
      m_is_stride_ceq = false;
    }
    this->m_size += strides[i] * heights[i];
  }

  AllocateDevice(ctx);
}

int CviImg::Init(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt) {
  this->m_fmt = fmt;
  this->m_channel = img_c;
  this->m_width = img_w;
  this->m_height = img_h;
  // FIXME: Think a method for kernels.
  // u32 w_mod = img_w % 16;
  u32 w_stride = img_w;  // (w_mod == 0) ? img_w : img_w + (16 - w_mod);
  this->m_strides.resize(this->m_channel, w_stride);
  this->m_heights.resize(this->m_channel, this->m_height);
  this->m_size = 1 * this->m_channel * this->m_height * w_stride * getFmtSize(this->m_fmt);
  if (this->m_fmt == FMT_U8) {
    if (this->m_channel == 1) {
      this->m_img_type = CVI_GRAY;
    } else if (this->m_channel == 3) {
      this->m_img_type = CVI_RGB_PLANAR;
    } else if (this->m_channel == 4) {
      this->m_img_type = CVI_RGBA_PLANAR;
    } else {
      this->m_img_type = CVI_MULTI;
    }
  } else {
    this->m_img_type = (this->m_channel == 1) ? CVI_SINGLE : CVI_MULTI;
  }
  this->m_is_planar = true;
  return AllocateDevice(ctx);
}

const bool CviImg::IsInit() { return this->m_bmmem == NULL ? false : true; }

uint8_t *CviImg::GetVAddr() { return m_vaddr; }

uint64_t CviImg::GetPAddr() const { return m_paddr; }

const u32 CviImg::GetImgChannel() const { return m_channel; }

const u32 CviImg::GetImgWidth() const { return m_width; }

const u32 CviImg::GetImgHeight() const { return m_height; }

const std::vector<u32> CviImg::GetImgStrides() const { return m_strides; }

const std::vector<u32> CviImg::GetImgHeights() const { return m_heights; }

const u64 CviImg::GetImgSize() const { return m_size; }

const bool CviImg::IsSubImg() const { return m_is_sub_img; }

const bool CviImg::IsStideCEQ() const { return m_is_stride_ceq; }

int CviImg::AllocateDevice(bmctx_t *ctx) {
  int ret = 1;
  if (this->m_bmmem == NULL) {
    ret = 0;
    this->m_bmmem = bmmem_device_alloc_raw(*ctx, this->m_size);

    if (m_is_stride_ceq) {
      this->m_tg.start_address = bmmem_device_addr(this->m_bmmem);
      this->m_tg.base_reg_index = 0;
      this->m_tg.fmt = this->m_fmt;
      this->m_tg.shape = {1, this->m_channel, this->m_height, this->m_width};
      this->m_tg.stride.h = this->m_strides[0] * getFmtSize(this->m_tg.fmt);
      this->m_tg.stride.c = m_tg.shape.h * this->m_tg.stride.h;
      this->m_tg.stride.n = m_tg.shape.c * this->m_tg.stride.c;
    }
  }
  m_vaddr = (uint8_t *)bmmem_device_v_addr(this->m_bmmem);
  m_paddr = bmmem_device_addr(this->m_bmmem);
  return ret;
}

int CviImg::Free(bmctx_t *ctx) {
  bmmem_device_free(*ctx, this->m_bmmem);
  return 0;
}