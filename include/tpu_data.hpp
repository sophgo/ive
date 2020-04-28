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
  int fmt_size = 0;
  switch (fmt) {
    case FMT_I8:
    case FMT_U8:
      fmt_size = 1;
      break;
    case FMT_U16:
    case FMT_I16:
    case FMT_BF16:
      fmt_size = 2;
      break;
    case FMT_U32:
    case FMT_I32:
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

enum CVIIMGTYPE {
  CVI_RGB_PLANAR = 0,
  CVI_RGB_PACKED,
  CVI_RGBA_PLANAR,
  CVI_GRAY,
  CVI_YUV420,
  CVI_YUV422,
  CVI_SINGLE,
  CVI_MULTI
};

inline bool IsImgPlanar(CVIIMGTYPE img_type) {
  bool is_planar = true;
  switch (img_type) {
    case CVI_RGB_PACKED:
    case CVI_YUV422:
      is_planar = false;
      break;
    default:
      break;
  }
  return is_planar;
}

/**
 * @brief A wrapper for TPU device memory defined in runtime.
 *        This is a class originally designed for TPU, so the default setup for image is planar.
 *        The sub-image constructor only supports planar images. If you want to create packed
 *        image, use the constructor with CVIIMGTYPE variable input.
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
  CviImg(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt, CviImg *cvi_img = nullptr);

  /**
   * @brief Construct a new CviImg object from an existing CviImg with given region.
   *
   * @param ctx bm context
   * @param img cvi_img
   */
  CviImg(bmctx_t *ctx, const CviImg &img, u32 x1, u32 y1, u32 x2, u32 y2);

  /**
   * @brief Construct a new CviImg object with given strides.
   *
   * @param ctx bm context
   * @param img_h Image height.
   * @param img_w Image width.
   * @param strides Image strides, the channel of CviImg will be set to the size of strides.
   * @param heights Image heights.
   * @param CVIIMGTYPE CviImg type enum.
   * @param fmt fmt_t type
   */
  CviImg(bmctx_t *ctx, u32 img_h, u32 img_w, std::vector<u32> strides, std::vector<u32> heights,
         CVIIMGTYPE img_type, fmt_t fmt, CviImg *cvi_img = nullptr);

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
  int Init(bmctx_t *ctx, u32 img_c, u32 img_h, u32 img_w, fmt_t fmt, CviImg *img_ptr);

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
   * @brief Get the channel of the image.
   *
   * @return const u32 image channel.
   */
  const u32 GetImgChannel() const;

  /**
   * @brief Get the width of the image.
   *
   * @return const u32 image width.
   */
  const u32 GetImgWidth() const;

  /**
   * @brief Get the height of the image.
   *
   * @return const u32 image height.
   */
  const u32 GetImgHeight() const;

  /**
   * @brief Get the strides of the image.
   *
   * @return const std::vector<u32> image strides.
   */
  const std::vector<u32> GetImgStrides() const;

  /**
   * @brief Get the heights of the image.
   *
   * @return const std::vector<u32> image heights.
   */
  const std::vector<u32> GetImgHeights() const;

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
   * @brief Tells if the image's strides differ bwtween channels.
   *
   * @return true All strides are the same.
   * @return false Not the same.
   */
  const bool IsStideCEQ() const;

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
   * @brief Setup image structure information. This mode does not support padding and packed mode.
   *
   * @param img_c Input image channel.
   * @param img_h Input image height.
   * @param img_w Input image width.
   * @param fmt fmt type.
   */
  inline void SetupImageInfo(u32 img_c, u32 img_h, u32 img_w, fmt_t fmt);

  /**
   * @brief Allocate device memory.
   *
   * @param ctx bm context
   * @return int Return 0 if success
   */
  int AllocateDevice(bmctx_t *ctx);

  u32 m_channel = 0;
  u32 m_width = 0;
  u32 m_height = 0;
  std::vector<u32> m_strides;
  std::vector<u32> m_heights;
  fmt_t m_fmt = FMT_U8;
  u64 m_size = 0;  // Total size of memory

  bmmem_device_t m_bmmem = NULL;  // Set to NULL if not initialized
  uint64_t m_paddr = -1;          // Set to maximum of uint64_t if not initaulized
  uint8_t *m_vaddr = nullptr;     // Set to nullptr if not initualized

  /**
   * These are variables used for different framework.
   * In order to know the format stored in CvImage from CSC, CVIIMAGETYPE is used. m_is_planar,
   * m_is_stride_ceq variables are used to check if TPU supports. As for m_is_sub_img, this
   * variable is used to check if ive should use channelExtension or not.
   */
  CVIIMGTYPE m_img_type;
  bool m_is_planar = true;      // Is image planar.
  bool m_is_sub_img = false;    // Is sub-image flag.
  bool m_is_stride_ceq = true;  // Are all the strides in every channel equal.
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