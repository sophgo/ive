#pragma once
#include "cvi_type.h"

#include <bmkernel/bm1880v2/bmkernel_1880v2.h>
#include <bmruntime.h>
#include <string.h>
#include <iostream>
#include <vector>

typedef struct cvi_chip_info {
  u32 version;
  u32 npu_num;
  u32 eu_num;
  u32 lmem_size;
  u32 lmem_banks;
  u32 lmem_bank_size;
} cvi_chip_info_s;

/**
 * @brief Convert fmt_t to actual data type size.
 *
 * @param fmt fmt_t defined in kernel
 * @return int Actual data type size
 */
static int getFmtSize(fmt_t fmt) {
  int fmt_size = 1;
  switch (fmt) {
    case FMT_I8:
    case FMT_U8:
      fmt_size = 1;
      break;
    case FMT_U16:
    case FMT_I16:
      fmt_size = 2;
      break;
    case FMT_BF16:
      fmt_size = 2;
      break;
    case FMT_F32:
      fmt_size = 4;
      break;
    default:
      std::cerr << "Unsupported fmt type: " << fmt << std::endl;
  }
  return fmt_size;
}

/**
 * @brief Information of a sliced image for one dimension.
 *
 */
struct sliceUnit {
  u32 slice;
  u32 skip;
  u32 turn;
  u32 left;
  u32 c_multiplier = 1;
};

/**
 * @brief Memory layout information for auto slicing function.
 *
 */
struct SliceInfo {
  fmt_t io_fmt = FMT_INVALID;
  u32 ping_pong_size = 1;
  u32 ping_pong_share_tl = 0;
  u32 nums_of_tl = 2;
  u32 fix_lmem_size = 0;
  u32 nums_of_table = 0;
};

struct SliceRes {
  sliceUnit h;
  sliceUnit w;
};

struct TLMemInfo {
  bmk1880v2_tensor_lmem_shape_t shape;
  bmk1880v2_tensor_lmem_stride_t stride;
};
struct TGMemInfo {
  bmk1880v2_tensor_tgmem_shape_t shape;
  bmk1880v2_tensor_tgmem_stride_t stride;
};
struct TensorSliceInfo {
  TLMemInfo tl_load;
  TLMemInfo tl_store;
  TGMemInfo tg_load;
  TGMemInfo tg_store;
};

/**
 * @brief Kernel information for IveCore base class.
 *
 */
struct kernelInfo {
  u32 nums_of_kernel = 0;
  bool use_multiplier = 0;
  u32 pad[4] = {0};  // L R T B
  u32 size = 1;
  u32 default_stride_x = 1;
  u32 default_stride_y = 1;
};

/**
 * @brief img_multiplier used by IveKernel.
 *
 */
struct img_multiplier {
  float f = 1.f;
  u32 base = 2147483647;
  int shift = 0;
};

/**
 * @brief A wrapper for TPU device memory defined in runtime.
 *
 */
class CviImg {
 public:
  /**
   * @brief Default constructor
   *
   */
  CviImg();
  /**
   * @brief Construct a new CviImg object
   *
   * @param ctx bm context
   * @param img_c Image channel
   * @param img_h Image height
   * @param img_w Image width
   * @param fmt fmt_t type
   */
  CviImg(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt);

  /**
   * @brief Construct a new Cvi Img object
   *
   * @param ctx bm context
   * @param img cvi_img
   */
  CviImg(bmctx_t *ctx, const CviImg &img, u32 x1, u32 y1, u32 x2, u32 y2);

  /**
   * @brief Construct a new CviImg object
   *
   * @param ctx bm context
   * @param img_c Image channel
   * @param img_h Image height
   * @param img_w Image width
   * @param fmt fmt_t type
   * @param m_bmmem Pre-allocate bmmem_device_t memory pointer
   */
  CviImg(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt, bmmem_device_t m_bmmem);

  /**
   * @brief Construct a new CviImg object
   *
   * @param ctx bm context
   * @param img_c Image channel
   * @param img_h Image height
   * @param img_w Image width
   * @param fmt fmt_t type
   * @param data Pre-allocate data array
   */
  CviImg(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt, u8 *data);

  /**
   * @brief Init CviImg if default constructor is used.
   *
   * @param ctx bm context
   * @param img_c Image channel
   * @param img_h Image height
   * @param img_w Image width
   * @param fmt fmt_t type
   * @return int Return 0 if success
   */
  int Init(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt);

  /**
   * @brief Check if device memory is allocated.
   *
   * @return true Device memory is allocated
   * @return false Deive memory is not allocated
   */
  const bool IsInit();

  /**
   * @brief Get virtual memory address from device memory.
   *
   * @return const uint8_t* Address pointer
   */
  uint8_t *GetVAddr();

  /**
   * @brief Get physical offset from device memory.
   *
   * @return const uint64_t Address offset
   */
  uint64_t GetPAddr() const;

  /**
   * @brief Get the size of the image in bytes.
   *
   * @return const u64 image size
   */
  const u64 GetImgSize() const;

  /**
   * @brief Tells if this image instance is a sub-image of an image.
   *
   * @return true Is sub-image.
   * @return false Not a sub-image.
   */
  const bool IsSubImg() const;

  /**
   * @brief Release allocated device memory.
   *
   * @param ctx bm context
   * @return int return 0 if success
   */
  int Free(bmctx_t *ctx);

  /**
   * @brief Flush cache data to RAM.
   *
   * @param ctx bm context.
   * @return int return 0 if success.
   */
  int Flush(bmctx_t *ctx) {
#ifdef CVI_SOC
    return bmmem_device_flush(*ctx, m_bmmem);
#else
    return 0;
#endif
  }

  /**
   * @brief Update cache data from RAM.
   *
   * @param ctx bm context.
   * @return int return 0 if success.
   */
  int Invld(bmctx_t *ctx) {
#ifdef CVI_SOC
    return bmmem_device_invld(*ctx, m_bmmem);
#else
    return 0;
#endif
  }

  bmk1880v2_tensor_tgmem_t m_tg;

 private:
  /**
   * @brief Allocate device memory.
   *
   * @param ctx bm context
   * @return int Return 0 if success
   */
  int AllocateDevice(bmctx_t *ctx);

  bmmem_device_t m_bmmem = NULL;  // Set to NULL if not initialized
  uint64_t m_paddr = -1;          // Set to maximum of uint64_t if not initaulized
  uint8_t *m_vaddr = nullptr;     // Set to nullptr if not initualized
  u64 m_size = 0;                 // Total size of memory
  bool m_is_sub_img = false;      // Is sub-image flag.
};

/**
 * @brief Ive kernel structure, for filter operations.
 *
 */
struct IveKernel {
  CviImg img;
  img_multiplier multiplier;
};

class FmtnSize {
 public:
  FmtnSize() {}
  FmtnSize(fmt_t fmt) { setFmt(fmt); }
  void setFmt(fmt_t fmt) {
    m_fmt = fmt;
    m_fmt_size = getFmtSize(m_fmt);
  }

  const fmt_t getFmt() const { return m_fmt; }
  const u32 getSize() const { return m_fmt_size; }

 private:
  fmt_t m_fmt = FMT_U8;
  u32 m_fmt_size = 1;
};

struct BMAddrInfo {
  std::vector<u64> addr_vec;
  std::vector<FmtnSize> fns_vec;
};

struct TLInfo {
  std::vector<bmk1880v2_tensor_lmem_t *> lmem_vec;
  std::vector<FmtnSize> fns_vec;
};