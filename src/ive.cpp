#include "ive.h"
#include "ive_experimental.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include "tracer/tracer.h"

#include "kernel_generator.hpp"
#include "table_manager.hpp"
#include "tpu_data.hpp"

#include "tpu/tpu_add.hpp"
#include "tpu/tpu_and.hpp"
#include "tpu/tpu_block.hpp"
#include "tpu/tpu_cmp.hpp"
#include "tpu/tpu_copy.hpp"
#include "tpu/tpu_filter.hpp"
#include "tpu/tpu_magandang.hpp"
#include "tpu/tpu_morph.hpp"
#include "tpu/tpu_mulsum.hpp"
#include "tpu/tpu_normalize.hpp"
#include "tpu/tpu_or.hpp"
#include "tpu/tpu_sad.hpp"
#include "tpu/tpu_sigmoid.hpp"
#include "tpu/tpu_sobel.hpp"
#include "tpu/tpu_sub.hpp"
#include "tpu/tpu_table.hpp"
#include "tpu/tpu_threshold.hpp"
#include "tpu/tpu_xor.hpp"

#include <stdarg.h>
#include <cmath>
#include <limits>

/**
 * @brief stringfy #define
 *
 * STRFY stringfy the variable itself.
 * VSTRFY stringfy the value saved in the variable.
 *
 */
#define STRFY(s) #s
#define VSTRFY(s) STRFY(s)

/**
 * @brief String array of IVE_IMAGE_S enType.
 *
 */
// clang-format off
const char *imgEnTypeStr[] = {STRFY(IVE_IMAGE_TYPE_U8C1),
                              STRFY(IVE_IMAGE_TYPE_S8C1),
                              STRFY(IVE_IMAGE_TYPE_YUV420SP),
                              STRFY(IVE_IMAGE_TYPE_YUV422SP),
                              STRFY(IVE_IMAGE_TYPE_YUV420P),
                              STRFY(IVE_IMAGE_TYPE_YUV422P),
                              STRFY(IVE_IMAGE_TYPE_S8C2_PACKAGE),
                              STRFY(IVE_IMAGE_TYPE_S8C2_PLANAR),
                              STRFY(IVE_IMAGE_TYPE_S16C1),
                              STRFY(IVE_IMAGE_TYPE_U16C1),
                              STRFY(IVE_IMAGE_TYPE_U8C3_PACKAGE),
                              STRFY(IVE_IMAGE_TYPE_U8C3_PLANAR),
                              STRFY(IVE_IMAGE_TYPE_S32C1),
                              STRFY(IVE_IMAGE_TYPE_U32C1),
                              STRFY(IVE_IMAGE_TYPE_S64C1),
                              STRFY(IVE_IMAGE_TYPE_U64C1),
                              STRFY(IVE_IMAGE_TYPE_BF16C1),
                              STRFY(IVE_IMAGE_TYPE_FP32C1)};
// clang-format on
// We use initializer_list to make sure the variadic input are correct.
namespace detail {
inline bool IsValidImageType(IVE_IMAGE_S *pstImg, std::string pstImgStr,
                             std::initializer_list<const IVE_IMAGE_TYPE_E> enType) {
  if (pstImg == NULL) {
    std::cerr << pstImgStr << " cannot be NULL." << std::endl;
    return false;
  }
  for (auto it : enType) {
    if (pstImg->enType == it) {
      return true;
    }
  }

  std::string msg = pstImgStr + " only supports ";
  for (auto it : enType) {
    msg += (std::string(imgEnTypeStr[it]) + std::string(" "));
  }
  std::cerr << msg << std::endl;
  return false;
}
}  // namespace detail

// The variadic function.
template <typename... Types>
inline bool IsValidImageType(IVE_IMAGE_S *pstImg, std::string pstImgStr, const Types... enType) {
  return detail::IsValidImageType(pstImg, pstImgStr, {enType...});
}

struct TPU_HANDLE {
  TblMgr t_tblmgr;
  IveTPUAdd t_add;
  IveTPUAddBF16 t_add_bf16;
  IveTPUAnd t_and;
  IveTPUBlock t_block;
  IveTPUBlockBF16 t_block_bf16;
  IveTPUCopyInterval t_copy_int;
  IveTPUErode t_erode;
  IveTPUFilter t_filter;
  IveTPUFilterBF16 t_filter_bf16;
  IveTPUMagAndAng t_magandang;
  IveTPUMax t_max;
  IveTPUMulSum t_mulsum;
  IveTPUMin t_min;
  IveTPUNormalize t_norm;
  IveTPUOr t_or;
  IveTPUSAD t_sad;
  IveTPUSigmoid t_sig;
  IveTPUSobelGradOnly t_sobel_gradonly;
  IveTPUSobel t_sobel;
  IveTPUSubAbs t_sub_abs;
  IveTPUSub t_sub;
  IveTPUTbl t_tbl;
  IveTPUThreshold t_thresh;
  IveTPUThresholdHighLow t_thresh_hl;
  IveTPUThresholdSlope t_thresh_s;
  IveTPUXOr t_xor;
};

struct IVE_HANDLE_CTX {
  bmctx_t ctx;
  cvk_context_t *cvk_ctx;
  TPU_HANDLE t_h;
  // VIP
};

IVE_HANDLE CVI_IVE_CreateHandle() {
  IVE_HANDLE_CTX *handle_ctx = new IVE_HANDLE_CTX;
  if (createHandle(&handle_ctx->ctx, &handle_ctx->cvk_ctx) != CVI_SUCCESS) {
    delete handle_ctx;
    return NULL;
  }
  if (handle_ctx->t_h.t_tblmgr.init(&handle_ctx->ctx, handle_ctx->cvk_ctx) != CVI_SUCCESS) {
    delete handle_ctx;
    return NULL;
  }
  return (void *)handle_ctx;
}

CVI_S32 CVI_IVE_DestroyHandle(IVE_HANDLE pIveHandle) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_tblmgr.free(&handle_ctx->ctx);
  destroyHandle(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  delete handle_ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_BufFlush(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  if (pstImg->tpu_block == NULL) {
    return CVI_FAILURE;
  }
  auto *img = reinterpret_cast<CviImg *>(pstImg->tpu_block);
  return img->Flush(&handle_ctx->ctx);
}

CVI_S32 CVI_IVE_BufRequest(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  if (pstImg->tpu_block == NULL) {
    return CVI_FAILURE;
  }
  auto *img = reinterpret_cast<CviImg *>(pstImg->tpu_block);
  return img->Invld(&handle_ctx->ctx);
}

CVI_S32 CVI_IVE_CmdFlush(IVE_HANDLE pIveHandle) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  cviruntime_cvikernel_submit(handle_ctx->ctx);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_CreateMemInfo(IVE_HANDLE pIveHandle, IVE_MEM_INFO_S *pstMemInfo,
                              CVI_U32 u32ByteSize) {
  pstMemInfo->u32PhyAddr = 0;
  pstMemInfo->pu8VirAddr = new CVI_U8[u32ByteSize];
  pstMemInfo->u32ByteSize = u32ByteSize;
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_CreateImage2(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg, IVE_IMAGE_TYPE_E enType,
                             u16 u16Width, u16 u16Height, IVE_IMAGE_S *pstBuffer) {
  if (u16Width == 0 || u16Height == 0) {
    std::cerr << "Image width or height cannot be 0." << std::endl;
    pstImg->tpu_block = NULL;
    pstImg->enType = enType;
    pstImg->u16Width = 0;
    pstImg->u16Height = 0;
    pstImg->u16Reserved = 0;
    for (size_t i = 0; i < 3; i++) {
      pstImg->pu8VirAddr[i] = NULL;
      pstImg->u64PhyAddr[i] = -1;
      pstImg->u16Stride[i] = 0;
    }
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  int fmt_size = 1;
  cvk_fmt_t fmt = CVK_FMT_U8;
  CVIIMGTYPE img_type;
  std::vector<u32> strides;
  std::vector<u32> heights;
  const u32 align = 16;
  switch (enType) {
    case IVE_IMAGE_TYPE_S8C1: {
      img_type = CVI_SINGLE;
      const u32 stride = WidthAlign(u16Width, align);
      strides.push_back(stride);
      heights.push_back(u16Height);
      fmt = CVK_FMT_I8;
    } break;
    case IVE_IMAGE_TYPE_U8C1: {
      const u32 stride = WidthAlign(u16Width, align);
      strides.push_back(stride);
      heights.push_back(u16Height);
      img_type = CVI_GRAY;
    } break;
    case IVE_IMAGE_TYPE_YUV420P: {
      img_type = CVI_YUV420;
      const u32 stride = WidthAlign(u16Width, align);
      strides.push_back(stride);
      const u32 stride2 = WidthAlign(u16Width >> 1, align);
      strides.push_back(stride2);
      strides.push_back(stride2);
      heights.push_back(u16Height);
      heights.push_back(u16Height >> 1);
      heights.push_back(u16Height >> 1);
    } break;
    case IVE_IMAGE_TYPE_YUV422P: {
      img_type = CVI_YUV422;
      const u32 stride = WidthAlign(u16Width, align);
      const u32 stride2 = WidthAlign(u16Width >> 1, align);
      strides.resize(1, stride);
      strides.resize(2, stride2);
      heights.resize(3, u16Height);
    } break;
    case IVE_IMAGE_TYPE_U8C3_PACKAGE: {
      img_type = CVI_RGB_PACKED;
      const u32 stride = WidthAlign(u16Width * 3, align);
      strides.push_back(stride);
      heights.push_back(u16Height);
    } break;
    case IVE_IMAGE_TYPE_U8C3_PLANAR: {
      img_type = CVI_RGB_PLANAR;
      const u32 stride = WidthAlign(u16Width, align);
      strides.resize(3, stride);
      heights.resize(3, u16Height);
    } break;
    case IVE_IMAGE_TYPE_BF16C1: {
      img_type = CVI_SINGLE;
      const u32 stride = WidthAlign(u16Width, align);
      strides.push_back(stride);
      heights.push_back(u16Height);
      fmt_size = 2;
      fmt = CVK_FMT_BF16;
    } break;
    case IVE_IMAGE_TYPE_U16C1: {
      img_type = CVI_SINGLE;
      const u32 stride = WidthAlign(u16Width, align);
      strides.push_back(stride);
      heights.push_back(u16Height);
      fmt_size = 2;
      fmt = CVK_FMT_U16;
    } break;
    case IVE_IMAGE_TYPE_S16C1: {
      img_type = CVI_SINGLE;
      const u32 stride = WidthAlign(u16Width, align);
      strides.push_back(stride);
      heights.push_back(u16Height);
      fmt_size = 2;
      fmt = CVK_FMT_I16;
    } break;
    case IVE_IMAGE_TYPE_U32C1: {
      img_type = CVI_SINGLE;
      const u32 stride = WidthAlign(u16Width, align);
      strides.push_back(stride);
      heights.push_back(u16Height);
      fmt_size = 4;
      fmt = CVK_FMT_U32;
    } break;
    case IVE_IMAGE_TYPE_FP32C1: {
      img_type = CVI_SINGLE;
      const u32 stride = WidthAlign(u16Width, align);
      strides.push_back(stride);
      heights.push_back(u16Height);
      fmt_size = 4;
      fmt = CVK_FMT_F32;
    } break;
    default:
      std::cerr << "Not supported enType " << imgEnTypeStr[enType] << std::endl;
      return CVI_FAILURE;
      break;
  }
  if (strides.size() == 0 || heights.size() == 0) {
    std::cerr << "[DEV] Stride not set." << std::endl;
    return CVI_FAILURE;
  }

  CviImg *buffer_ptr =
      pstBuffer == NULL ? nullptr : reinterpret_cast<CviImg *>(pstBuffer->tpu_block);
  auto *cpp_img = new CviImg(&handle_ctx->ctx, u16Height, u16Width, strides, heights, img_type, fmt,
                             buffer_ptr);
  if (!cpp_img->IsInit()) {
    return CVI_FAILURE;
  }

  pstImg->tpu_block = reinterpret_cast<CVI_IMG *>(cpp_img);
  pstImg->enType = enType;
  pstImg->u16Width = cpp_img->GetImgWidth();
  pstImg->u16Height = cpp_img->GetImgHeight();
  pstImg->u16Reserved = fmt_size;

  size_t i_limit = cpp_img->GetImgChannel();
  for (size_t i = 0; i < i_limit; i++) {
    pstImg->pu8VirAddr[i] = cpp_img->GetVAddr() + cpp_img->GetImgCOffsets()[i];
    pstImg->u64PhyAddr[i] = cpp_img->GetPAddr() + cpp_img->GetImgCOffsets()[i];
    pstImg->u16Stride[i] = cpp_img->GetImgStrides()[i];
  }

  for (size_t i = i_limit; i < 3; i++) {
    pstImg->pu8VirAddr[i] = NULL;
    pstImg->u64PhyAddr[i] = -1;
    pstImg->u16Stride[i] = 0;
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_CreateImage(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg, IVE_IMAGE_TYPE_E enType,
                            CVI_U16 u16Width, CVI_U16 u16Height) {
  return CVI_IVE_CreateImage2(pIveHandle, pstImg, enType, u16Width, u16Height, NULL);
}

CVI_S32 CVI_IVE_SubImage(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                         CVI_U16 u16X1, CVI_U16 u16Y1, CVI_U16 u16X2, CVI_U16 u16Y2) {
  if (u16X1 >= u16X2 || u16Y1 >= u16Y2) {
    std::cerr << "(X1, Y1) must smaller than (X2, Y2)." << std::endl;
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  auto *src_img = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  auto *cpp_img = new CviImg(&handle_ctx->ctx, *src_img, u16X1, u16Y1, u16X2, u16Y2);
  if (cpp_img->GetVAddr() == nullptr) {
    delete cpp_img;
    return CVI_FAILURE;
  }
  pstDst->tpu_block = reinterpret_cast<CVI_IMG *>(cpp_img);

  pstDst->enType = pstSrc->enType;
  pstDst->u16Width = cpp_img->m_tg.shape.w;
  pstDst->u16Height = cpp_img->m_tg.shape.h;
  pstDst->u16Reserved = pstSrc->u16Reserved;

  for (size_t i = 0; i < cpp_img->m_tg.shape.c; i++) {
    pstDst->pu8VirAddr[i] = cpp_img->GetVAddr() + cpp_img->GetImgCOffsets()[i];
    pstDst->u64PhyAddr[i] = cpp_img->GetPAddr() + cpp_img->GetImgCOffsets()[i];
    pstDst->u16Stride[i] = cpp_img->GetImgStrides()[i];
  }

  for (size_t i = cpp_img->m_tg.shape.c; i < 3; i++) {
    pstDst->pu8VirAddr[i] = NULL;
    pstDst->u64PhyAddr[i] = -1;
    pstDst->u16Stride[i] = 0;
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Image2VideoFrame(IVE_IMAGE_S *pstIISrc, VIDEO_FRAME_S *pstVFDst) {
  memset(pstVFDst, 0, sizeof(VIDEO_FRAME_S));
  switch (pstIISrc->enType) {
    case IVE_IMAGE_TYPE_U8C1: {
      pstVFDst->enPixelFormat = PIXEL_FORMAT_YUV_400;
    } break;
    case IVE_IMAGE_TYPE_YUV420P: {
      pstVFDst->enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
    } break;
    case IVE_IMAGE_TYPE_YUV422P: {
      pstVFDst->enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_422;
    } break;
    case IVE_IMAGE_TYPE_U8C3_PACKAGE: {
      pstVFDst->enPixelFormat = PIXEL_FORMAT_RGB_888;
    } break;
    case IVE_IMAGE_TYPE_U8C3_PLANAR: {
      pstVFDst->enPixelFormat = PIXEL_FORMAT_RGB_888_PLANAR;
    } break;
    default: {
      std::cerr << "Unsupported conversion type: " << imgEnTypeStr[pstIISrc->enType] << std::endl;
      return CVI_FAILURE;
    } break;
  }
  auto *src_img = reinterpret_cast<CviImg *>(pstIISrc->tpu_block);
  pstVFDst->u32Width = pstIISrc->u16Width;
  pstVFDst->u32Height = pstIISrc->u16Height;
  for (size_t i = 0; i < src_img->GetImgHeights().size(); i++) {
    pstVFDst->u32Stride[i] = pstIISrc->u16Stride[i];
    pstVFDst->u64PhyAddr[i] = pstIISrc->u64PhyAddr[i];
    pstVFDst->pu8VirAddr[i] = pstIISrc->pu8VirAddr[i];
  }
  for (size_t i = 0; i < src_img->GetImgHeights().size() - 1; i++) {
    pstVFDst->u32Length[i] = src_img->GetImgCOffsets()[i + 1];
  }
  if (src_img->GetImgHeights().size() > 2) {
    pstVFDst->u32Length[2] = src_img->GetImgSize() - src_img->GetImgCOffsets()[2];
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_VideoFrame2Image(VIDEO_FRAME_S *pstVFSrc, IVE_IMAGE_S *pstIIDst) {
  size_t c = 1;
  CVIIMGTYPE img_type = CVIIMGTYPE::CVI_GRAY;
  cvk_fmt_t fmt = CVK_FMT_U8;
  std::vector<u32> heights;
  switch (pstVFSrc->enPixelFormat) {
    case PIXEL_FORMAT_YUV_400: {
      pstIIDst->enType = IVE_IMAGE_TYPE_U8C1;
      heights.push_back(pstVFSrc->u32Height);
    } break;
    case PIXEL_FORMAT_YUV_PLANAR_420: {
      c = 3;
      img_type = CVIIMGTYPE::CVI_YUV420;
      pstIIDst->enType = IVE_IMAGE_TYPE_YUV420P;
      heights.push_back(pstVFSrc->u32Height);
      heights.push_back(pstVFSrc->u32Height >> 1);
      heights.push_back(pstVFSrc->u32Height >> 1);
    } break;
    case PIXEL_FORMAT_YUV_PLANAR_422: {
      c = 3;
      img_type = CVIIMGTYPE::CVI_YUV422;
      pstIIDst->enType = IVE_IMAGE_TYPE_YUV422P;
      heights.resize(3, pstVFSrc->u32Height);
    } break;
    case PIXEL_FORMAT_RGB_888: {
      c = 1;
      img_type = CVIIMGTYPE::CVI_RGB_PACKED;
      pstIIDst->enType = IVE_IMAGE_TYPE_U8C3_PACKAGE;
      heights.push_back(pstVFSrc->u32Height);
    } break;
    case PIXEL_FORMAT_RGB_888_PLANAR: {
      c = 3;
      img_type = CVIIMGTYPE::CVI_RGB_PLANAR;
      pstIIDst->enType = IVE_IMAGE_TYPE_U8C3_PLANAR;
      heights.resize(c, pstVFSrc->u32Height);
    } break;
    default: {
      std::cerr << "Unsupported conversion type: " << pstVFSrc->enPixelFormat << std::endl;
      return CVI_FAILURE;
    } break;
  }
  std::vector<u32> strides, u32_length;
  for (size_t i = 0; i < c; i++) {
    strides.push_back(pstVFSrc->u32Stride[i]);
    u32_length.push_back(pstVFSrc->u32Length[i]);
  }
  auto *cpp_img = new CviImg(pstVFSrc->u32Height, pstVFSrc->u32Width, strides, heights, u32_length,
                             pstVFSrc->pu8VirAddr[0], pstVFSrc->u64PhyAddr[0], img_type, fmt);
  if (!cpp_img->IsInit()) {
    return CVI_FAILURE;
  }

  pstIIDst->tpu_block = reinterpret_cast<CVI_IMG *>(cpp_img);
  pstIIDst->u16Width = cpp_img->GetImgWidth();
  pstIIDst->u16Height = cpp_img->GetImgHeight();
  pstIIDst->u16Reserved = getFmtSize(fmt);

  size_t i_limit = cpp_img->GetImgChannel();
  for (size_t i = 0; i < i_limit; i++) {
    pstIIDst->pu8VirAddr[i] = cpp_img->GetVAddr() + cpp_img->GetImgCOffsets()[i];
    pstIIDst->u64PhyAddr[i] = cpp_img->GetPAddr() + cpp_img->GetImgCOffsets()[i];
    pstIIDst->u16Stride[i] = cpp_img->GetImgStrides()[i];
  }

  for (size_t i = i_limit; i < 3; i++) {
    pstIIDst->pu8VirAddr[i] = NULL;
    pstIIDst->u64PhyAddr[i] = -1;
    pstIIDst->u16Stride[i] = 0;
  }
  return CVI_SUCCESS;
}

IVE_IMAGE_S CVI_IVE_ReadImage(IVE_HANDLE pIveHandle, const char *filename,
                              IVE_IMAGE_TYPE_E enType) {
  int desiredNChannels = -1;
  switch (enType) {
    case IVE_IMAGE_TYPE_U8C1:
      desiredNChannels = STBI_grey;
      break;
    case IVE_IMAGE_TYPE_U8C3_PLANAR:
      desiredNChannels = STBI_rgb;
      break;
    case IVE_IMAGE_TYPE_U8C3_PACKAGE:
      desiredNChannels = STBI_rgb;
      break;
    default:
      std::cerr << "Not support channel " << imgEnTypeStr[enType] << std::endl;
      break;
  }
  IVE_IMAGE_S img;
  if (desiredNChannels >= 0) {
    int width, height, nChannels;
    stbi_uc *stbi_data = stbi_load(filename, &width, &height, &nChannels, desiredNChannels);
    CVI_IVE_CreateImage(pIveHandle, &img, enType, width, height);
    if (stbi_data == nullptr) {
      std::cerr << "Image " << filename << " read failed." << std::endl;
      return img;
    }
    printf("desiredNChannels, width, height: %d %d %d\n", desiredNChannels, width, height);
    if (enType == IVE_IMAGE_TYPE_U8C3_PLANAR) {
      for (size_t i = 0; i < (size_t)height; i++) {
        for (size_t j = 0; j < (size_t)width; j++) {
          size_t stb_idx = (i * width + j) * 3;
          size_t img_idx = (i * img.u16Stride[0] + j);
          img.pu8VirAddr[0][img_idx] = stbi_data[stb_idx];
          img.pu8VirAddr[1][img_idx] = stbi_data[stb_idx + 1];
          img.pu8VirAddr[2][img_idx] = stbi_data[stb_idx + 2];
        }
      }
    } else {
      stbi_uc *ptr = stbi_data;
      for (size_t i = 0; i < (size_t)desiredNChannels; i++) {
        for (size_t j = 0; j < (size_t)height; j++) {
          memcpy(img.pu8VirAddr[i] + (j * img.u16Stride[i]), ptr, width);
          ptr += width;
        }
      }
    }
    CVI_IVE_BufFlush(pIveHandle, &img);
    stbi_image_free(stbi_data);
  }
  return img;
}

CVI_S32 CVI_IVE_WriteImage(IVE_HANDLE pIveHandle, const char *filename, IVE_IMAGE_S *pstImg) {
  int desiredNChannels = -1;
  int stride = 1;
  uint8_t *arr = nullptr;
  bool remove_buffer = false;
  switch (pstImg->enType) {
    case IVE_IMAGE_TYPE_U8C1:
      desiredNChannels = STBI_grey;
      arr = pstImg->pu8VirAddr[0];
      break;
    case IVE_IMAGE_TYPE_U8C3_PLANAR: {
      desiredNChannels = STBI_rgb;
      stride = 1;
      arr = new uint8_t[pstImg->u16Stride[0] * pstImg->u16Height * desiredNChannels];
      size_t image_total = pstImg->u16Stride[0] * pstImg->u16Height;
      for (size_t i = 0; i < image_total; i++) {
        size_t stb_idx = i * 3;
        arr[stb_idx] = pstImg->pu8VirAddr[0][i];
        arr[stb_idx + 1] = pstImg->pu8VirAddr[1][i];
        arr[stb_idx + 2] = pstImg->pu8VirAddr[2][i];
      }
      stride = 3;
      remove_buffer = true;
    } break;
    case IVE_IMAGE_TYPE_U8C3_PACKAGE:
      desiredNChannels = STBI_rgb;
      arr = pstImg->pu8VirAddr[0];
      stride = 3;
      break;
    default:
      std::cerr << "Not support channel " << imgEnTypeStr[pstImg->enType] << std::endl;
      return CVI_FAILURE;
      break;
  }
  CVI_IVE_BufRequest(pIveHandle, pstImg);
  stbi_write_png(filename, pstImg->u16Width, pstImg->u16Height, desiredNChannels, arr,
                 pstImg->u16Stride[0] * stride);
  if (remove_buffer) {
    delete[] arr;
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_SYS_FreeM(IVE_HANDLE pIveHandle, IVE_MEM_INFO_S *pstMemInfo) {
  delete[] pstMemInfo->pu8VirAddr;
  pstMemInfo->u32ByteSize = 0;
  return CVI_SUCCESS;
}

CVI_S32 CVI_SYS_FreeI(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  auto *cpp_img = reinterpret_cast<CviImg *>(pstImg->tpu_block);
  cpp_img->Free(&handle_ctx->ctx);
  delete cpp_img;
  cpp_img = nullptr;
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_DMA(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                    IVE_DMA_CTRL_S *pstDmaCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  int ret = CVI_FAILURE;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  if (pstDmaCtrl->enMode == IVE_DMA_MODE_DIRECT_COPY) {
#ifdef USE_CPU_COPY
    CVI_IVE_BufRequest(pIveHandle, pstSrc);
    CVI_IVE_BufRequest(pIveHandle, pstDst);
    uint size = pstSrc->u16Stride[0] * pstSrc->u16Height;
    memcpy(pstDst->pu8VirAddr[0], pstSrc->pu8VirAddr[0], size);
    CVI_IVE_BufFlush(pIveHandle, pstSrc);
    CVI_IVE_BufFlush(pIveHandle, pstDst);
#else
    CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
    CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
    std::vector<CviImg> inputs = {*cpp_src};
    std::vector<CviImg> outputs = {*cpp_dst};

    ret = IveTPUCopyDirect::run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
#endif
  } else if (pstDmaCtrl->enMode == IVE_DMA_MODE_INTERVAL_COPY) {
    handle_ctx->t_h.t_copy_int.setInvertal(pstDmaCtrl->u8HorSegSize, pstDmaCtrl->u8VerSegRows);
    handle_ctx->t_h.t_copy_int.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
    CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
    std::vector<CviImg> inputs = {*cpp_src};
    std::vector<CviImg> outputs = {*cpp_dst};

    ret = handle_ctx->t_h.t_copy_int.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs,
                                         true);
  }
  return ret;
}

CVI_S32 CVI_IVE_ImageTypeConvert(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc,
                                 IVE_DST_IMAGE_S *pstDst, IVE_ITC_CRTL_S *pstItcCtrl,
                                 bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  if (pstItcCtrl->enType == IVE_ITC_NORMALIZE) {
    if (cpp_src->m_tg.fmt == CVK_FMT_BF16 && cpp_dst->m_tg.fmt == CVK_FMT_F32) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      float *dst_ptr = (float *)cpp_dst->GetVAddr();
      u64 img_size = cpp_src->GetImgSize() / 2;
      neonBF162F32(src_ptr, dst_ptr, img_size);
      cpp_src->Flush(&handle_ctx->ctx);
      cpp_dst->Flush(&handle_ctx->ctx);
    } else if (cpp_src->m_tg.fmt == CVK_FMT_BF16 && cpp_dst->m_tg.fmt == CVK_FMT_U16) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      u16 *dst_ptr = (u16 *)cpp_dst->GetVAddr();
      float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
      u64 img_size = cpp_src->m_tg.shape.c * cpp_src->m_tg.shape.h * cpp_src->m_tg.shape.w;
      neonBF16FindMinMax(src_ptr, img_size, &min, &max);
      neonBF162U16Normalize(src_ptr, dst_ptr, img_size, min, max);
      cpp_src->Flush(&handle_ctx->ctx);
      cpp_dst->Flush(&handle_ctx->ctx);
    } else if (cpp_src->m_tg.fmt == CVK_FMT_BF16 && cpp_dst->m_tg.fmt == CVK_FMT_I16) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      s16 *dst_ptr = (s16 *)cpp_dst->GetVAddr();
      float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
      u64 img_size = cpp_src->m_tg.shape.c * cpp_src->m_tg.shape.h * cpp_src->m_tg.shape.w;
      neonBF16FindMinMax(src_ptr, img_size, &min, &max);
      neonBF162S16Normalize(src_ptr, dst_ptr, img_size, min, max);
      cpp_src->Flush(&handle_ctx->ctx);
      cpp_dst->Flush(&handle_ctx->ctx);
    } else if (cpp_src->m_tg.fmt == CVK_FMT_U16 &&
               (cpp_dst->m_tg.fmt == CVK_FMT_U8 || cpp_dst->m_tg.fmt == CVK_FMT_I8)) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      u8 *dst_ptr = (u8 *)cpp_dst->GetVAddr();
      u16 min = 65535, max = 0;
      u64 img_size = cpp_src->m_tg.shape.c * cpp_src->m_tg.shape.h * cpp_src->m_tg.shape.w;
      neonU16FindMinMax(src_ptr, img_size, &min, &max);
      if (cpp_dst->m_tg.fmt == CVK_FMT_U8) {
        neonU162U8Normalize(src_ptr, dst_ptr, img_size, min, max);
      } else {
        neonU162S8Normalize(src_ptr, (s8 *)dst_ptr, img_size, min, max);
      }
      cpp_src->Flush(&handle_ctx->ctx);
      cpp_dst->Flush(&handle_ctx->ctx);
    } else if (cpp_src->m_tg.fmt == CVK_FMT_BF16 &&
               (cpp_dst->m_tg.fmt == CVK_FMT_U8 || cpp_dst->m_tg.fmt == CVK_FMT_I8)) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
      u64 img_size = cpp_src->m_tg.shape.c * cpp_src->m_tg.shape.h * cpp_src->m_tg.shape.w;
      neonBF16FindMinMax(src_ptr, img_size, &min, &max);
      handle_ctx->t_h.t_norm.setMinMax(min, max);
      handle_ctx->t_h.t_norm.setOutputFMT(cpp_dst->m_tg.fmt);
      handle_ctx->t_h.t_norm.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
      CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
      CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
      std::vector<CviImg> inputs = {*cpp_src};
      std::vector<CviImg> outputs = {*cpp_dst};
      handle_ctx->t_h.t_norm.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
    }
  } else if (pstItcCtrl->enType == IVE_ITC_SATURATE) {
    if (cpp_src->m_tg.fmt == CVK_FMT_BF16 && cpp_dst->m_tg.fmt == CVK_FMT_F32) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      float *dst_ptr = (float *)cpp_dst->GetVAddr();
      u64 img_size = cpp_src->GetImgSize() / 2;
      neonBF162F32(src_ptr, dst_ptr, img_size);
      cpp_src->Flush(&handle_ctx->ctx);
      cpp_dst->Flush(&handle_ctx->ctx);
    } else if (cpp_src->m_tg.fmt == CVK_FMT_BF16 && cpp_dst->m_tg.fmt == CVK_FMT_U16) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      u16 *dst_ptr = (u16 *)cpp_dst->GetVAddr();
      u64 img_size = cpp_src->GetImgSize() / 2;
      neonBF162U16(src_ptr, dst_ptr, img_size);
      cpp_src->Flush(&handle_ctx->ctx);
      cpp_dst->Flush(&handle_ctx->ctx);
    } else if (cpp_src->m_tg.fmt == CVK_FMT_BF16 && cpp_dst->m_tg.fmt == CVK_FMT_I16) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      s16 *dst_ptr = (s16 *)cpp_dst->GetVAddr();
      u64 img_size = cpp_src->GetImgSize() / 2;
      neonBF162S16(src_ptr, dst_ptr, img_size);
      cpp_src->Flush(&handle_ctx->ctx);
      cpp_dst->Flush(&handle_ctx->ctx);
    } else if ((cpp_src->m_tg.fmt == CVK_FMT_BF16 || cpp_src->m_tg.fmt == CVK_FMT_U8 ||
                cpp_src->m_tg.fmt == CVK_FMT_I8) &&
               (cpp_dst->m_tg.fmt == CVK_FMT_BF16 || cpp_dst->m_tg.fmt == CVK_FMT_U8 ||
                cpp_dst->m_tg.fmt == CVK_FMT_I8)) {
      CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
      CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
      std::vector<CviImg> inputs = {*cpp_src};
      std::vector<CviImg> outputs = {*cpp_dst};

      IveTPUCopyDirect::run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
    } else {
      std::cerr << "Unsupported input output image type ( " << cpp_src->m_tg.fmt << ", "
                << cpp_dst->m_tg.fmt << ")." << std::endl;
      return CVI_FAILURE;
    }
  } else {
    std::cerr << "Unsupported enType " << pstItcCtrl->enType << std::endl;
    return CVI_FAILURE;
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Add(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, IVE_ADD_CTRL_S *ctrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc1, STRFY(pstSrc1), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR,
                        IVE_IMAGE_TYPE_BF16C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstSrc2, STRFY(pstSrc2), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR,
                        IVE_IMAGE_TYPE_BF16C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR,
                        IVE_IMAGE_TYPE_BF16C1)) {
    return CVI_FAILURE;
  }
  int ret = CVI_FAILURE;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs = {*cpp_dst};

  const float &x = ctrl->aX;
  const float &y = ctrl->bY;
  bool is_bf16 =
      (pstSrc1->enType == IVE_IMAGE_TYPE_BF16C1 || pstSrc2->enType == IVE_IMAGE_TYPE_BF16C1 ||
       pstDst->enType == IVE_IMAGE_TYPE_BF16C1)
          ? true
          : false;
  if (((x == 1 && y == 1) || (x == 0.f && y == 0.f)) || !is_bf16) {
    handle_ctx->t_h.t_add.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    ret = handle_ctx->t_h.t_add.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
  } else {
    handle_ctx->t_h.t_add_bf16.setCoef(x, y);
    handle_ctx->t_h.t_add_bf16.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    ret = handle_ctx->t_h.t_add_bf16.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
  }
  return ret;
}

CVI_S32 CVI_IVE_And(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc1, STRFY(pstSrc1), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstSrc2, STRFY(pstSrc2), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_and.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs = {*cpp_dst};

  return handle_ctx->t_h.t_and.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
}

CVI_S32 CVI_IVE_BLOCK(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                      IVE_BLOCK_CTRL_S *pstBlkCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR,
                        IVE_IMAGE_TYPE_BF16C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR,
                        IVE_IMAGE_TYPE_BF16C1)) {
    return CVI_FAILURE;
  }
  CVI_U32 u32CellSize = pstBlkCtrl->u32CellSize;
  if (pstDst->u16Width != (pstSrc->u16Width / u32CellSize) ||
      (pstSrc->u16Width % u32CellSize != 0)) {
    std::cerr << "Dst block width not match! Src: " << pstSrc->u16Width
              << ", dst: " << pstDst->u16Width << ", cell size: " << u32CellSize << std::endl;
    return CVI_FAILURE;
  }
  if (pstDst->u16Height != (pstSrc->u16Height / u32CellSize) ||
      (pstSrc->u16Height % u32CellSize != 0)) {
    std::cerr << "Dst block height not match! Src: " << pstSrc->u16Height
              << ", dst: " << pstDst->u16Height << ", cell size: " << u32CellSize << std::endl;
    return CVI_FAILURE;
  }

  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};
  int ret = CVI_FAILURE;
  if (cpp_src->m_tg.fmt == CVK_FMT_U8 && cpp_dst->m_tg.fmt == CVK_FMT_U8) {
    handle_ctx->t_h.t_block.setBinNum(pstBlkCtrl->f32BinSize);
    handle_ctx->t_h.t_block.setCellSize(u32CellSize, cpp_src->m_tg.shape.c);
    handle_ctx->t_h.t_block.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    ret =
        handle_ctx->t_h.t_block.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs, true);
  } else {
    handle_ctx->t_h.t_block_bf16.setBinNum(pstBlkCtrl->f32BinSize);
    handle_ctx->t_h.t_block_bf16.setCellSize(u32CellSize, cpp_src->m_tg.shape.c);
    handle_ctx->t_h.t_block_bf16.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    ret = handle_ctx->t_h.t_block_bf16.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs,
                                           true);
  }
  return ret;
}

CVI_S32 CVI_IVE_Dilate(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_DILATE_CTRL_S *pstDilateCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_filter.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};

  u32 npu_num = handle_ctx->t_h.t_erode.getNpuNum(handle_ctx->cvk_ctx);
  CviImg cimg(&handle_ctx->ctx, npu_num, 5, 5, CVK_FMT_U8);
  IveKernel kernel;
  kernel.img = cimg;
  kernel.img.GetVAddr();
  for (size_t i = 0; i < npu_num; i++) {
    memcpy(kernel.img.GetVAddr() + i * 25, pstDilateCtrl->au8Mask, 25);
  }
  kernel.multiplier.f = 1.f;
  QuantizeMultiplierSmallerThanOne(kernel.multiplier.f, &kernel.multiplier.base,
                                   &kernel.multiplier.shift);
  handle_ctx->t_h.t_filter.setKernel(kernel);
  int ret = handle_ctx->t_h.t_filter.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
  kernel.img.Free(&handle_ctx->ctx);
  return ret;
}

CVI_S32 CVI_IVE_Erode(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                      IVE_ERODE_CTRL_S *pstErodeCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_erode.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};

  u32 npu_num = handle_ctx->t_h.t_erode.getNpuNum(handle_ctx->cvk_ctx);
  CviImg cimg(&handle_ctx->ctx, npu_num, 5, 5, CVK_FMT_U8);
  IveKernel kernel;
  kernel.img = cimg;
  kernel.img.GetVAddr();
  for (size_t i = 0; i < npu_num; i++) {
    memcpy(kernel.img.GetVAddr() + i * 25, pstErodeCtrl->au8Mask, 25);
  }
  kernel.multiplier.f = 1.f;
  QuantizeMultiplierSmallerThanOne(kernel.multiplier.f, &kernel.multiplier.base,
                                   &kernel.multiplier.shift);
  handle_ctx->t_h.t_erode.setKernel(kernel);
  int ret = handle_ctx->t_h.t_erode.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
  kernel.img.Free(&handle_ctx->ctx);
  return ret;
}

CVI_S32 CVI_IVE_Filter(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_FILTER_CTRL_S *pstFltCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  if (pstSrc->enType != pstDst->enType) {
    std::cerr << "pstSrc & pstDst must have the same type." << std::endl;
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_filter.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};

  if (pstFltCtrl->u8MaskSize != 3 && pstFltCtrl->u8MaskSize != 5) {
    std::cerr << "Currently Filter only supports filter size 3 or 5." << std::endl;
  }
  u32 npu_num = handle_ctx->t_h.t_filter.getNpuNum(handle_ctx->cvk_ctx);
  CviImg cimg(&handle_ctx->ctx, npu_num, pstFltCtrl->u8MaskSize, pstFltCtrl->u8MaskSize,
              CVK_FMT_I8);
  IveKernel kernel;
  kernel.img = cimg;
  kernel.img.GetVAddr();
  int mask_length = pstFltCtrl->u8MaskSize * pstFltCtrl->u8MaskSize;
  for (size_t i = 0; i < npu_num; i++) {
    memcpy((s8 *)(kernel.img.GetVAddr() + i * mask_length), pstFltCtrl->as8Mask, mask_length);
  }
  kernel.img.Flush(&handle_ctx->ctx);
  kernel.multiplier.f = 1.f / pstFltCtrl->u32Norm;
  QuantizeMultiplierSmallerThanOne(kernel.multiplier.f, &kernel.multiplier.base,
                                   &kernel.multiplier.shift);
  handle_ctx->t_h.t_filter.setKernel(kernel);
  int ret = handle_ctx->t_h.t_filter.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
  kernel.img.Free(&handle_ctx->ctx);
  return ret;
}

inline bool get_hog_feature_info(u16 width, u16 height, u16 u32CellSize, u16 blk_size,
                                 u32 *width_cell, u32 *height_cell, u32 *width_block,
                                 u32 *height_block) {
  *height_cell = (u32)height / u32CellSize;
  *width_cell = (u32)width / u32CellSize;
  if (*height_cell < blk_size || *width_cell < blk_size) {
    return false;
  }
  *height_block = (*height_cell - blk_size) + 1;
  *width_block = (*width_cell - blk_size) + 1;
  return true;
}

CVI_S32 CVI_IVE_GET_HOG_SIZE(CVI_U16 u16Width, CVI_U16 u16Height, CVI_U8 u8BinSize,
                             CVI_U16 u16CellSize, CVI_U16 u16BlkSizeInCell, CVI_U16 u16BlkStepX,
                             CVI_U16 u16BlkStepY, CVI_U32 *u32HogSize) {
  if (u16BlkStepX == 0) {
    std::cerr << "u16BlkStepX cannot be 0." << std::endl;
    return CVI_FAILURE;
  }
  if (u16BlkStepY == 0) {
    std::cerr << "u16BlkStepX cannot be 0." << std::endl;
    return CVI_FAILURE;
  }
  u32 height_cell = 0, width_cell = 0, height_block = 0, width_block = 0;
  if (!get_hog_feature_info(u16Width, u16Height, u16CellSize, u16BlkSizeInCell, &width_cell,
                            &height_cell, &width_block, &height_block)) {
    std::cerr << "Block size exceed cell block." << std::endl;
    return CVI_FAILURE;
  }
  u32 block_length = u16BlkSizeInCell * u16BlkSizeInCell;
  width_block = (width_block - 1) / u16BlkStepX + 1;
  height_block = (height_block - 1) / u16BlkStepY + 1;
  u32 num_of_block_data = height_block * width_block;
  *u32HogSize = num_of_block_data * (block_length * u8BinSize) * sizeof(float);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_HOG(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDstH,
                    IVE_DST_IMAGE_S *pstDstV, IVE_DST_IMAGE_S *pstDstMag,
                    IVE_DST_IMAGE_S *pstDstAng, IVE_DST_MEM_INFO_S *pstDstHist,
                    IVE_HOG_CTRL_S *pstHogCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  // No need to check here. Will check later.
  if (pstDstAng->u16Width % pstHogCtrl->u32CellSize != 0) {
    std::cerr << "Width " << pstDstAng->u16Width << " is not divisible by "
              << pstHogCtrl->u32CellSize << std::endl;
    return CVI_FAILURE;
  }
  if (pstDstAng->u16Height % pstHogCtrl->u32CellSize != 0) {
    std::cerr << "Height " << pstDstAng->u16Height << " is not divisible by "
              << pstHogCtrl->u32CellSize << std::endl;
    return CVI_FAILURE;
  }
  if (pstHogCtrl->u16BlkStepX == 0) {
    std::cerr << "u16BlkStepX cannot be 0." << std::endl;
    return CVI_FAILURE;
  }
  if (pstHogCtrl->u16BlkStepY == 0) {
    std::cerr << "u16BlkStepX cannot be 0." << std::endl;
    return CVI_FAILURE;
  }
  u32 height_cell = 0, width_cell = 0, height_block = 0, width_block = 0;
  if (!get_hog_feature_info(pstDstAng->u16Width, pstDstAng->u16Height, pstHogCtrl->u32CellSize,
                            pstHogCtrl->u16BlkSizeInCell, &width_cell, &height_cell, &width_block,
                            &height_block)) {
    std::cerr << "Block size exceed cell block." << std::endl;
    return CVI_FAILURE;
  }
  // u32 &&cell_length = pstHogCtrl->u32CellSize * pstHogCtrl->u32CellSize;
  u32 &&cell_hist_length = height_cell * width_cell * pstHogCtrl->u8BinSize;
  u32 &&block_length = pstHogCtrl->u16BlkSizeInCell * pstHogCtrl->u16BlkSizeInCell;
  u32 &&num_of_block_data = ((height_block - 1) / pstHogCtrl->u16BlkStepY + 1) *
                            ((width_block - 1) / pstHogCtrl->u16BlkStepX + 1);
  u32 &&hog_hist_length = num_of_block_data * (block_length * pstHogCtrl->u8BinSize);
  u32 &&hog_hist_size = hog_hist_length * sizeof(float);
  if (pstDstHist->u32ByteSize != hog_hist_size) {
    std::cerr << "Histogram size not match! Given: " << pstDstHist->u32ByteSize
              << ", required: " << hog_hist_size << " (" << hog_hist_length << " * sizeof(u32))"
              << std::endl;
    return CVI_FAILURE;
  }

  IVE_SOBEL_CTRL_S iveSblCtrl;
  iveSblCtrl.enOutCtrl = IVE_SOBEL_OUT_CTRL_BOTH;
  iveSblCtrl.u8MaskSize = 1;
  if (CVI_IVE_Sobel(pIveHandle, pstSrc, pstDstH, pstDstV, &iveSblCtrl, 0) != CVI_SUCCESS) {
    return CVI_FAILURE;
  }
  IVE_MAG_AND_ANG_CTRL_S iveMaaCtrl;
  iveMaaCtrl.enOutCtrl = IVE_MAG_AND_ANG_OUT_CTRL_MAG_AND_ANG;
  iveMaaCtrl.enDistCtrl = IVE_MAG_DIST_L2;
  if (CVI_IVE_MagAndAng(pIveHandle, pstDstH, pstDstV, pstDstMag, pstDstAng, &iveMaaCtrl, 0) !=
      CVI_SUCCESS) {
    return CVI_FAILURE;
  }

  // Get Histogram here.
  Tracer::TraceBegin("CPU histogram");
  Tracer::TraceBegin("Generate cell histogram");
  CVI_IVE_BufRequest(pIveHandle, pstDstAng);
  CVI_IVE_BufRequest(pIveHandle, pstDstMag);
  u16 *ang_ptr = (u16 *)pstDstAng->pu8VirAddr[0];
  u16 *mag_ptr = (u16 *)pstDstMag->pu8VirAddr[0];
  float div = 180 / pstHogCtrl->u8BinSize;
  float *cell_histogram = new float[cell_hist_length];
  memset(cell_histogram, 0, cell_hist_length * sizeof(float));
  // Do Add & DIV while creating histogram. Slow.
  auto &&u16Height_i = (u32)(pstDstAng->u16Height - 1);
  auto &&u16Width_j = (u32)(pstDstAng->u16Width - 1);
  for (u32 i = 1; i < u16Height_i; i++) {
    u32 &&row_skip = pstDstAng->u16Stride[0] * i;
    u32 &&cell_row_skip = (i / pstHogCtrl->u32CellSize) * width_cell;
    for (u32 j = 1; j < u16Width_j; j++) {
      u32 &&cell_index =
          (u32)(cell_row_skip + (u32)(j / pstHogCtrl->u32CellSize)) * pstHogCtrl->u8BinSize;
      u32 degree = std::abs(convert_bf16_fp32(ang_ptr[j + row_skip]));
      u32 mag = convert_bf16_fp32(mag_ptr[j + row_skip]);
      float bin_div = degree / div;
      float bin_div_dec = bin_div - (u32)(bin_div);
      if (bin_div_dec == 0) {
        u32 bin_index = bin_div;
        if (bin_index == pstHogCtrl->u8BinSize) {
          bin_index = 0;
        }
        cell_histogram[cell_index + bin_index] += mag;
      } else {
        u32 bin_index = bin_div;
        if (bin_index == pstHogCtrl->u8BinSize) {
          bin_index = 0;
        }
        u32 bin_index_2 = (bin_index + 1);
        if (bin_index_2 >= pstHogCtrl->u8BinSize) bin_index_2 = 0;
        float bin_div_dec_left = 1.f - bin_div_dec;
        cell_histogram[cell_index + bin_index] += (mag * bin_div_dec_left);
        cell_histogram[cell_index + bin_index_2] += (mag * bin_div_dec);
      }
    }
  }
  Tracer::TraceEnd();

  Tracer::TraceBegin("Generate HOG histogram");
  u32 &&copy_length = pstHogCtrl->u8BinSize * pstHogCtrl->u16BlkSizeInCell;
  u32 &&copy_data_length = copy_length * sizeof(float);
  float *hog_ptr = (float *)pstDstHist->pu8VirAddr;
  memset(hog_ptr, 0, pstDstHist->u32ByteSize);
  u32 count = 0;
  for (u32 i = 0; i < height_block; i += pstHogCtrl->u16BlkStepY) {
    u32 &&row_skip = i * width_block;
    for (u32 j = 0; j < width_block; j += pstHogCtrl->u16BlkStepX) {
      u32 &&skip = j + row_skip;
      for (u32 k = 0; k < pstHogCtrl->u16BlkSizeInCell; k++) {
        u32 &&index = skip + k * width_block;
        auto *cell_hist_ptr = cell_histogram + index;
        auto *dst_hog_ptr = hog_ptr + count;
        memcpy(dst_hog_ptr, cell_hist_ptr, copy_data_length);
        count += copy_length;
      }
    }
  }
  delete[] cell_histogram;
  if (count != hog_hist_length) {
    std::cerr << "Hog histogram not aligned." << count << " " << hog_hist_length << std::endl;
    return CVI_FAILURE;
  }
  Tracer::TraceEnd();
  Tracer::TraceBegin("Normalizing HOG histogram");
  hog_ptr = (float *)pstDstHist->pu8VirAddr;
  u32 &&block_data_length = block_length * pstHogCtrl->u8BinSize;
  u32 nums_of_block_feature = hog_hist_length / block_data_length;
#ifdef __ARM_ARCH_7A__
  const u32 neon_turn = 0;
#else
  u32 neon_turn = block_data_length / 4;
#endif
  u32 neon_turn_left = neon_turn * 4;
  for (u32 i = 0; i < nums_of_block_feature; i++) {
    float count_total = 0;
    auto &&skip_i = i * block_data_length;
    float *block_head = hog_ptr + skip_i;
#ifndef __ARM_ARCH_7A__
    for (u32 j = 0; j < neon_turn; j++) {
      float32x4_t f = vld1q_f32(block_head);
      float32x4_t result = vmulq_f32(f, f);
      count_total += vaddvq_f32(result);
      block_head += 4;
    }
#endif
    for (u32 j = neon_turn_left; j < block_data_length; j++) {
      count_total += hog_ptr[skip_i + j] * hog_ptr[skip_i + j];
    }
    float count = count_total == 0 ? 0 : 1.f / sqrt(count_total);
    float32x4_t m = vdupq_n_f32(count);
    block_head = hog_ptr + skip_i;
    for (u32 j = 0; j < neon_turn; j++) {
      float32x4_t f = vld1q_f32(block_head);
      float32x4_t result = vmulq_f32(m, f);
      vst1q_f32(block_head, result);
      block_head += 4;
    }
    for (u32 j = neon_turn_left; j < block_data_length; j++) {
      hog_ptr[skip_i + j] *= count;
    }
  }
  Tracer::TraceEnd();
  Tracer::TraceEnd();
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_MagAndAng(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrcH, IVE_SRC_IMAGE_S *pstSrcV,
                          IVE_DST_IMAGE_S *pstDstMag, IVE_DST_IMAGE_S *pstDstAng,
                          IVE_MAG_AND_ANG_CTRL_S *pstMaaCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrcH, STRFY(pstSrcH), IVE_IMAGE_TYPE_BF16C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstSrcV, STRFY(pstSrcV), IVE_IMAGE_TYPE_BF16C1)) {
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_magandang.setTblMgr(&handle_ctx->t_h.t_tblmgr);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrcH->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrcV->tpu_block);
  CviImg *cpp_dst =
      (pstDstMag != NULL) ? reinterpret_cast<CviImg *>(pstDstMag->tpu_block) : nullptr;
  CviImg *cpp_dst2 =
      (pstDstAng != NULL) ? reinterpret_cast<CviImg *>(pstDstAng->tpu_block) : nullptr;
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs;
  switch (pstMaaCtrl->enOutCtrl) {
    case IVE_MAG_AND_ANG_OUT_CTRL_MAG: {
      if (!IsValidImageType(pstDstMag, STRFY(pstDstMag), IVE_IMAGE_TYPE_BF16C1)) {
        return CVI_FAILURE;
      }
      handle_ctx->t_h.t_magandang.exportOption(true, false);
      outputs.emplace_back(*cpp_dst);
    } break;
    case IVE_MAG_AND_ANG_OUT_CTRL_ANG: {
      if (!IsValidImageType(pstDstAng, STRFY(pstDstAng), IVE_IMAGE_TYPE_BF16C1)) {
        return CVI_FAILURE;
      }
      handle_ctx->t_h.t_magandang.exportOption(false, true);
      outputs.emplace_back(*cpp_dst2);
    } break;
    case IVE_MAG_AND_ANG_OUT_CTRL_MAG_AND_ANG: {
      if (!IsValidImageType(pstDstMag, STRFY(pstDstMag), IVE_IMAGE_TYPE_BF16C1)) {
        return CVI_FAILURE;
      }
      if (!IsValidImageType(pstDstAng, STRFY(pstDstAng), IVE_IMAGE_TYPE_BF16C1)) {
        return CVI_FAILURE;
      }
      handle_ctx->t_h.t_magandang.exportOption(true, true);
      outputs.emplace_back(*cpp_dst);
      outputs.emplace_back(*cpp_dst2);
    } break;
    default:
      std::cerr << "Not supported Mag and Angle type." << std::endl;
      return CVI_FAILURE;
      break;
  }
  // True accuracy too low.
  handle_ctx->t_h.t_magandang.noNegative(false);
  handle_ctx->t_h.t_magandang.magDistMethod(pstMaaCtrl->enDistCtrl);
  handle_ctx->t_h.t_magandang.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  return handle_ctx->t_h.t_magandang.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
}

CVI_S32 CVI_IVE_Map(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_MEM_INFO_S *pstMap,
                    IVE_DST_IMAGE_S *pstDst, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  auto &shape = handle_ctx->t_h.t_tblmgr.getTblTLShape(CVK_FMT_U8);
  u32 tbl_sz = shape.h * shape.w;
  if (pstMap->u32ByteSize != tbl_sz) {
    std::cerr << "Mapping table must be size " << tbl_sz << " in CVI_U8 format." << std::endl;
    return CVI_FAILURE;
  }
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};
  handle_ctx->t_h.t_tbl.setTable(&handle_ctx->ctx, &handle_ctx->t_h.t_tblmgr, pstMap->pu8VirAddr);
  handle_ctx->t_h.t_tbl.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  return handle_ctx->t_h.t_tbl.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
}

CVI_S32 CVI_IVE_MulSum(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg, double *sum, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstImg, STRFY(pstImg), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_BF16C1)) {
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstImg->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs;
  handle_ctx->t_h.t_mulsum.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  int ret = handle_ctx->t_h.t_mulsum.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
  *sum = handle_ctx->t_h.t_mulsum.getSum();
  return ret;
}

CVI_S32 CVI_IVE_NormGrad(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDstH,
                         IVE_DST_IMAGE_S *pstDstV, IVE_DST_IMAGE_S *pstDstHV,
                         IVE_NORM_GRAD_CTRL_S *pstNormGradCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  int kernel_size = pstNormGradCtrl->u8MaskSize;
  if (kernel_size != 1 && kernel_size != 3) {
    std::cerr << "Kernel size currently only supports 1 and 3." << std::endl;
    return CVI_FAILURE;
  }

  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  int npu_num = handle_ctx->t_h.t_sobel_gradonly.getNpuNum(handle_ctx->cvk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs;
  bool do_free = false;
  int ret = CVI_FAILURE;
  if (pstNormGradCtrl->enOutCtrl == IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER) {
    if (!IsValidImageType(pstDstH, STRFY(pstDstH), IVE_IMAGE_TYPE_S16C1, IVE_IMAGE_TYPE_U8C1)) {
      return CVI_FAILURE;
    }
    if (!IsValidImageType(pstDstV, STRFY(pstDstV), IVE_IMAGE_TYPE_S16C1, IVE_IMAGE_TYPE_U8C1)) {
      return CVI_FAILURE;
    }
    IVE_IMAGE_S dstH_BF16, dstV_BF16;
    CVI_IVE_CreateImage(pIveHandle, &dstH_BF16, IVE_IMAGE_TYPE_BF16C1, pstSrc->u16Width,
                        pstSrc->u16Height);
    CVI_IVE_CreateImage(pIveHandle, &dstV_BF16, IVE_IMAGE_TYPE_BF16C1, pstSrc->u16Width,
                        pstSrc->u16Height);
    CviImg *cpp_dstv = reinterpret_cast<CviImg *>(dstV_BF16.tpu_block);
    CviImg *cpp_dsth = reinterpret_cast<CviImg *>(dstH_BF16.tpu_block);
    outputs.emplace_back(*cpp_dstv);
    outputs.emplace_back(*cpp_dsth);
    IveKernel kernel_w =
        createKernel(&handle_ctx->ctx, npu_num, kernel_size, kernel_size, IVE_KERNEL::SOBEL_X);
    IveKernel kernel_h =
        createKernel(&handle_ctx->ctx, npu_num, kernel_size, kernel_size, IVE_KERNEL::SOBEL_Y);
    handle_ctx->t_h.t_sobel_gradonly.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    handle_ctx->t_h.t_sobel_gradonly.setKernel(kernel_w, kernel_h);
    ret = handle_ctx->t_h.t_sobel_gradonly.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs,
                                               &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
    kernel_h.img.Free(&handle_ctx->ctx);
    IVE_ITC_CRTL_S iveItcCtrl;
    iveItcCtrl.enType = pstNormGradCtrl->enITCType;
    ret |= CVI_IVE_ImageTypeConvert(pIveHandle, &dstV_BF16, pstDstV, &iveItcCtrl, 0);
    ret |= CVI_IVE_ImageTypeConvert(pIveHandle, &dstH_BF16, pstDstH, &iveItcCtrl, 0);
    ret |= CVI_SYS_FreeI(pIveHandle, &dstV_BF16);
    ret |= CVI_SYS_FreeI(pIveHandle, &dstH_BF16);
  } else if (pstNormGradCtrl->enOutCtrl == IVE_NORM_GRAD_OUT_CTRL_HOR) {
    if (!IsValidImageType(pstDstH, STRFY(pstDstH), IVE_IMAGE_TYPE_S16C1, IVE_IMAGE_TYPE_U8C1)) {
      return CVI_FAILURE;
    }
    IVE_IMAGE_S dst_BF16;
    if (pstDstH->enType == IVE_IMAGE_TYPE_U16C1 ||
        pstNormGradCtrl->enITCType == IVE_ITC_NORMALIZE) {
      CVI_IVE_CreateImage(pIveHandle, &dst_BF16, IVE_IMAGE_TYPE_BF16C1, pstSrc->u16Width,
                          pstSrc->u16Height);
      CviImg *cpp_dsth = reinterpret_cast<CviImg *>(dst_BF16.tpu_block);
      outputs.emplace_back(*cpp_dsth);
      do_free = true;
    } else {
      CviImg *cpp_dsth = reinterpret_cast<CviImg *>(pstDstH->tpu_block);
      outputs.emplace_back(*cpp_dsth);
    }
    IveKernel kernel_h =
        createKernel(&handle_ctx->ctx, npu_num, kernel_size, kernel_size, IVE_KERNEL::SOBEL_Y);
    handle_ctx->t_h.t_filter_bf16.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    handle_ctx->t_h.t_filter_bf16.setKernel(kernel_h);
    ret =
        handle_ctx->t_h.t_filter_bf16.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
    kernel_h.img.Free(&handle_ctx->ctx);
    if (do_free) {
      IVE_ITC_CRTL_S iveItcCtrl;
      iveItcCtrl.enType = pstNormGradCtrl->enITCType;
      ret |= CVI_IVE_ImageTypeConvert(pIveHandle, &dst_BF16, pstDstH, &iveItcCtrl, 0);
      ret |= CVI_SYS_FreeI(pIveHandle, &dst_BF16);
    }
  } else if (pstNormGradCtrl->enOutCtrl == IVE_NORM_GRAD_OUT_CTRL_VER) {
    if (!IsValidImageType(pstDstV, STRFY(pstDstV), IVE_IMAGE_TYPE_S16C1, IVE_IMAGE_TYPE_U8C1)) {
      return CVI_FAILURE;
    }
    IVE_IMAGE_S dst_BF16;
    if (pstDstV->enType == IVE_IMAGE_TYPE_U16C1 ||
        pstNormGradCtrl->enITCType == IVE_ITC_NORMALIZE) {
      CVI_IVE_CreateImage(pIveHandle, &dst_BF16, IVE_IMAGE_TYPE_BF16C1, pstSrc->u16Width,
                          pstSrc->u16Height);
      CviImg *cpp_dstv = reinterpret_cast<CviImg *>(dst_BF16.tpu_block);
      outputs.emplace_back(*cpp_dstv);
      do_free = true;
    } else {
      CviImg *cpp_dstv = reinterpret_cast<CviImg *>(pstDstV->tpu_block);
      outputs.emplace_back(*cpp_dstv);
    }
    IveKernel kernel_w =
        createKernel(&handle_ctx->ctx, npu_num, kernel_size, kernel_size, IVE_KERNEL::SOBEL_X);
    handle_ctx->t_h.t_filter_bf16.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    handle_ctx->t_h.t_filter_bf16.setKernel(kernel_w);
    ret =
        handle_ctx->t_h.t_filter_bf16.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
    if (do_free) {
      IVE_ITC_CRTL_S iveItcCtrl;
      iveItcCtrl.enType = pstNormGradCtrl->enITCType;
      ret |= CVI_IVE_ImageTypeConvert(pIveHandle, &dst_BF16, pstDstV, &iveItcCtrl, 0);
      ret |= CVI_SYS_FreeI(pIveHandle, &dst_BF16);
    }
  } else if (pstNormGradCtrl->enOutCtrl == IVE_NORM_GRAD_OUT_CTRL_COMBINE) {
    if (!IsValidImageType(pstDstHV, STRFY(pstDstHV), IVE_IMAGE_TYPE_U16C1, IVE_IMAGE_TYPE_U8C1)) {
      return CVI_FAILURE;
    }
    IVE_IMAGE_S dst_BF16;
    if (pstDstHV->enType == IVE_IMAGE_TYPE_U16C1 ||
        pstNormGradCtrl->enITCType == IVE_ITC_NORMALIZE) {
      CVI_IVE_CreateImage(pIveHandle, &dst_BF16, IVE_IMAGE_TYPE_BF16C1, pstSrc->u16Width,
                          pstSrc->u16Height);
      CviImg *cpp_dsthv = reinterpret_cast<CviImg *>(dst_BF16.tpu_block);
      outputs.emplace_back(*cpp_dsthv);
      do_free = true;
    } else {
      CviImg *cpp_dsthv = reinterpret_cast<CviImg *>(pstDstHV->tpu_block);
      outputs.emplace_back(*cpp_dsthv);
    }
    IveKernel kernel_w =
        createKernel(&handle_ctx->ctx, npu_num, kernel_size, kernel_size, IVE_KERNEL::SOBEL_X);
    IveKernel kernel_h =
        createKernel(&handle_ctx->ctx, npu_num, kernel_size, kernel_size, IVE_KERNEL::SOBEL_Y);

    handle_ctx->t_h.t_sobel.setTblMgr(&handle_ctx->t_h.t_tblmgr);
    handle_ctx->t_h.t_sobel.magDistMethod(pstNormGradCtrl->enDistCtrl);
    handle_ctx->t_h.t_sobel.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    handle_ctx->t_h.t_sobel.setKernel(kernel_w, kernel_h);
    ret = handle_ctx->t_h.t_sobel.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
    kernel_h.img.Free(&handle_ctx->ctx);
    if (do_free) {
      IVE_ITC_CRTL_S iveItcCtrl;
      iveItcCtrl.enType = pstNormGradCtrl->enITCType;
      ret |= CVI_IVE_ImageTypeConvert(pIveHandle, &dst_BF16, pstDstHV, &iveItcCtrl, 0);
      ret |= CVI_SYS_FreeI(pIveHandle, &dst_BF16);
    }
  } else {
    return ret;
  }
  return ret;
}

CVI_S32 CVI_IVE_Or(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                   IVE_DST_IMAGE_S *pstDst, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc1, STRFY(pstSrc1), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstSrc2, STRFY(pstSrc2), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_or.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs = {*cpp_dst};

  return handle_ctx->t_h.t_or.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
}

CVI_S32 CVI_IVE_OrdStatFilter(IVE_HANDLE *pIveHandle, IVE_SRC_IMAGE_S *pstSrc,
                              IVE_DST_IMAGE_S *pstDst,
                              IVE_ORD_STAT_FILTER_CTRL_S *pstOrdStatFltCtrl, bool bInstant) {
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  const u32 kz = 3;
  const u32 pad_sz = kz - 1;
  if ((pstDst->u16Width + pad_sz != pstSrc->u16Width) ||
      (pstDst->u16Height + pad_sz != pstSrc->u16Height)) {
    std::cerr << "Error, pstDst (width, height) should be pstSrc (width - " << pad_sz
              << ", height - " << pad_sz << ")." << std::endl;
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};
  int ret = CVI_SUCCESS;
  if (pstOrdStatFltCtrl->enMode == IVE_ORD_STAT_FILTER_MODE_MAX) {
    handle_ctx->t_h.t_max.setKernelSize(kz);
    handle_ctx->t_h.t_max.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    ret |= handle_ctx->t_h.t_max.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
  } else if (pstOrdStatFltCtrl->enMode == IVE_ORD_STAT_FILTER_MODE_MIN) {
    handle_ctx->t_h.t_min.setKernelSize(kz);
    handle_ctx->t_h.t_min.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    ret |= handle_ctx->t_h.t_min.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
  }
  return ret;
}

CVI_S32 CVI_IVE_Sigmoid(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                        bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_BF16C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_BF16C1)) {
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_add.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};
  handle_ctx->t_h.t_sig.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  return handle_ctx->t_h.t_sig.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
}

CVI_S32 CVI_IVE_SAD(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstSad, IVE_DST_IMAGE_S *pstThr, IVE_SAD_CTRL_S *pstSadCtrl,
                    bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc1, STRFY(pstSrc1), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstSrc2, STRFY(pstSrc2), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstSad, STRFY(pstSad), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U16C1,
                        IVE_IMAGE_TYPE_S16C1, IVE_IMAGE_TYPE_BF16C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstThr, STRFY(pstThr), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  if (pstSrc1->u16Width != pstSrc2->u16Width || pstSrc1->u16Height != pstSrc2->u16Height) {
    std::cerr << "Two input size must be the same!" << std::endl;
    return CVI_FAILURE;
  }
  CVI_U32 window_size = 1;
  switch (pstSadCtrl->enMode) {
    case IVE_SAD_MODE_MB_4X4:
      window_size = 4;
      break;
    case IVE_SAD_MODE_MB_8X8:
      window_size = 8;
      break;
    case IVE_SAD_MODE_MB_16X16:
      window_size = 16;
      break;
    default:
      std::cerr << "Unsupported SAD mode " << pstSadCtrl->enMode << std::endl;
      return CVI_FAILURE;
      break;
  }
  if (pstSad->u16Width != pstSrc1->u16Width) {
    std::cerr << "Dst width not match with src! Src: " << pstSrc1->u16Width
              << ", dst: " << pstSad->u16Width << std::endl;
    return CVI_FAILURE;
  }
  if (pstSad->u16Height != pstSrc1->u16Height) {
    std::cerr << "Dst height not match with src! Src: " << pstSrc1->u16Height
              << ", dst: " << pstSad->u16Height << std::endl;
    return CVI_FAILURE;
  }
  int ret = CVI_SUCCESS;
  bool do_threshold = (pstSadCtrl->enOutCtrl == IVE_SAD_OUT_CTRL_16BIT_BOTH ||
                       pstSadCtrl->enOutCtrl == IVE_SAD_OUT_CTRL_8BIT_BOTH ||
                       pstSadCtrl->enOutCtrl == IVE_SAD_OUT_CTRL_THRESH)
                          ? true
                          : false;
  bool is_output_u8 = (pstSadCtrl->enOutCtrl == IVE_SAD_OUT_CTRL_8BIT_SAD ||
                       pstSadCtrl->enOutCtrl == IVE_SAD_OUT_CTRL_8BIT_BOTH ||
                       pstSadCtrl->enOutCtrl == IVE_SAD_OUT_CTRL_THRESH)
                          ? true
                          : false;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
  CviImg *cpp_dst = nullptr;
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs;
  IVE_IMAGE_S dst_BF16;
  if (!is_output_u8) {
    ret = CVI_IVE_CreateImage(pIveHandle, &dst_BF16, IVE_IMAGE_TYPE_BF16C1, pstSad->u16Width,
                              pstSad->u16Height);
    cpp_dst = reinterpret_cast<CviImg *>(dst_BF16.tpu_block);
  } else {
    cpp_dst = reinterpret_cast<CviImg *>(pstSad->tpu_block);
  }
  if (do_threshold) {
    if (pstSad->u16Width != pstThr->u16Width || pstSad->u16Height != pstThr->u16Height) {
      std::cerr << "Threshold output size must be the same as SAD output!" << std::endl;
      return CVI_FAILURE;
    }
    CviImg *thresh_dst = reinterpret_cast<CviImg *>(pstThr->tpu_block);
    if (pstSadCtrl->enOutCtrl == IVE_SAD_OUT_CTRL_THRESH) {
      outputs.emplace_back(*thresh_dst);
      handle_ctx->t_h.t_sad.outputThresholdOnly(true);
    } else {
      outputs.emplace_back(*cpp_dst);
      outputs.emplace_back(*thresh_dst);
    }
  } else {
    outputs.emplace_back(*cpp_dst);
  }
  handle_ctx->t_h.t_sad.setTblMgr(&handle_ctx->t_h.t_tblmgr);
  handle_ctx->t_h.t_sad.doThreshold(do_threshold);
  handle_ctx->t_h.t_sad.setThreshold(pstSadCtrl->u16Thr, pstSadCtrl->u8MinVal,
                                     pstSadCtrl->u8MaxVal);
  handle_ctx->t_h.t_sad.setWindowSize(window_size);
  handle_ctx->t_h.t_sad.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  ret = handle_ctx->t_h.t_sad.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
  if (!is_output_u8) {
    IVE_ITC_CRTL_S iveItcCtrl;
    iveItcCtrl.enType = IVE_ITC_SATURATE;
    ret |= CVI_IVE_ImageTypeConvert(pIveHandle, &dst_BF16, pstSad, &iveItcCtrl, 0);
    ret |= CVI_SYS_FreeI(pIveHandle, &dst_BF16);
  }
  return ret;
}

CVI_S32 CVI_IVE_Sobel(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDstH,
                      IVE_DST_IMAGE_S *pstDstV, IVE_SOBEL_CTRL_S *pstSobelCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_sobel.setTblMgr(&handle_ctx->t_h.t_tblmgr);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dsth = reinterpret_cast<CviImg *>(pstDstH->tpu_block);
  CviImg *cpp_dstv = reinterpret_cast<CviImg *>(pstDstV->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs;
  u8 mask_sz = pstSobelCtrl->u8MaskSize;
  int ret = CVI_FAILURE;
  if (pstSobelCtrl->enOutCtrl == IVE_SOBEL_OUT_CTRL_BOTH) {
    if (!IsValidImageType(pstDstH, STRFY(pstDstH), IVE_IMAGE_TYPE_BF16C1)) {
      return CVI_FAILURE;
    }
    if (!IsValidImageType(pstDstV, STRFY(pstDstV), IVE_IMAGE_TYPE_BF16C1)) {
      return CVI_FAILURE;
    }
    int npu_num = handle_ctx->t_h.t_sobel_gradonly.getNpuNum(handle_ctx->cvk_ctx);
    outputs.emplace_back(*cpp_dstv);
    outputs.emplace_back(*cpp_dsth);
    IveKernel kernel_w =
        createKernel(&handle_ctx->ctx, npu_num, mask_sz, mask_sz, IVE_KERNEL::SOBEL_X);
    IveKernel kernel_h =
        createKernel(&handle_ctx->ctx, npu_num, mask_sz, mask_sz, IVE_KERNEL::SOBEL_Y);
    handle_ctx->t_h.t_sobel_gradonly.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    handle_ctx->t_h.t_sobel_gradonly.setKernel(kernel_w, kernel_h);
    ret = handle_ctx->t_h.t_sobel_gradonly.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs,
                                               &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
    kernel_h.img.Free(&handle_ctx->ctx);
  } else if (pstSobelCtrl->enOutCtrl == IVE_SOBEL_OUT_CTRL_HOR) {
    if (!IsValidImageType(pstDstH, STRFY(pstDstH), IVE_IMAGE_TYPE_BF16C1)) {
      return CVI_FAILURE;
    }
    outputs.emplace_back(*cpp_dsth);
    IveKernel kernel_h = createKernel(&handle_ctx->ctx, cpp_src->m_tg.shape.c, mask_sz, mask_sz,
                                      IVE_KERNEL::SOBEL_Y);
    handle_ctx->t_h.t_filter_bf16.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    handle_ctx->t_h.t_filter_bf16.setKernel(kernel_h);
    ret =
        handle_ctx->t_h.t_filter_bf16.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
    kernel_h.img.Free(&handle_ctx->ctx);
  } else if (pstSobelCtrl->enOutCtrl == IVE_SOBEL_OUT_CTRL_VER) {
    if (!IsValidImageType(pstDstV, STRFY(pstDstV), IVE_IMAGE_TYPE_BF16C1)) {
      return CVI_FAILURE;
    }
    outputs.emplace_back(*cpp_dstv);
    IveKernel kernel_w = createKernel(&handle_ctx->ctx, cpp_src->m_tg.shape.c, mask_sz, mask_sz,
                                      IVE_KERNEL::SOBEL_X);
    handle_ctx->t_h.t_filter_bf16.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    handle_ctx->t_h.t_filter_bf16.setKernel(kernel_w);
    ret =
        handle_ctx->t_h.t_filter_bf16.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
  } else {
    return ret;
  }
  return ret;
}

CVI_S32 CVI_IVE_Sub(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, IVE_SUB_CTRL_S *ctrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc1, STRFY(pstSrc1), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstSrc2, STRFY(pstSrc2), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  int ret = CVI_FAILURE;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  if (ctrl->enMode == IVE_SUB_MODE_NORMAL) {
    handle_ctx->t_h.t_sub.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
    CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
    CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
    std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
    std::vector<CviImg> outputs = {*cpp_dst};

    ret = handle_ctx->t_h.t_sub.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
  } else if (ctrl->enMode == IVE_SUB_MODE_ABS) {
    handle_ctx->t_h.t_sub_abs.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
    CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
    CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
    CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
    std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
    std::vector<CviImg> outputs = {*cpp_dst};

    ret = handle_ctx->t_h.t_sub_abs.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
  }
  return ret;
}

CVI_S32 CVI_IVE_Thresh(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_THRESH_CTRL_S *ctrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  int ret = CVI_FAILURE;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};

  if (ctrl->enMode == IVE_THRESH_MODE_BINARY) {
    if (ctrl->u8MinVal == 0 && ctrl->u8MaxVal == 255) {
      handle_ctx->t_h.t_thresh.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
      handle_ctx->t_h.t_thresh.setThreshold(ctrl->u8LowThr);
      ret = handle_ctx->t_h.t_thresh.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
    } else {
      handle_ctx->t_h.t_thresh_hl.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
      handle_ctx->t_h.t_thresh_hl.setThreshold(ctrl->u8LowThr, ctrl->u8MinVal, ctrl->u8MaxVal);
      ret =
          handle_ctx->t_h.t_thresh_hl.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
    }
  }
  return ret;
}

CVI_S32 CVI_IVE_Thresh_S16(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                           IVE_THRESH_S16_CTRL_S *pstThrS16Ctrl, bool bInstant) {
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_S16C1)) {
    return CVI_FAILURE;
  }
  CVI_IVE_BufRequest(pIveHandle, pstSrc);
  CVI_IVE_BufRequest(pIveHandle, pstDst);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  u64 data_size = cpp_src->m_tg.stride.n / getFmtSize(cpp_src->m_tg.fmt);
  if (pstThrS16Ctrl->enMode == IVE_THRESH_S16_MODE_S16_TO_S8_MIN_MID_MAX ||
      pstThrS16Ctrl->enMode == IVE_THRESH_S16_MODE_S16_TO_S8_MIN_ORI_MAX) {
    if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_S8C1)) {
      return CVI_FAILURE;
    }
    bool is_mmm =
        (pstThrS16Ctrl->enMode == IVE_THRESH_S16_MODE_S16_TO_S8_MIN_MID_MAX) ? true : false;
    neonS162S8ThresholdLH((s16 *)pstSrc->pu8VirAddr[0], (s8 *)pstDst->pu8VirAddr[0], data_size,
                          pstThrS16Ctrl->s16LowThr, pstThrS16Ctrl->s16HighThr,
                          pstThrS16Ctrl->un8MinVal.s8Val, pstThrS16Ctrl->un8MidVal.s8Val,
                          pstThrS16Ctrl->un8MaxVal.s8Val, is_mmm);
  } else {
    if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1)) {
      return CVI_FAILURE;
    }
    bool is_mmm =
        (pstThrS16Ctrl->enMode == IVE_THRESH_S16_MODE_S16_TO_U8_MIN_MID_MAX) ? true : false;
    neonS162U8ThresholdLH((s16 *)pstSrc->pu8VirAddr[0], (u8 *)pstDst->pu8VirAddr[0], data_size,
                          pstThrS16Ctrl->s16LowThr, pstThrS16Ctrl->s16HighThr,
                          pstThrS16Ctrl->un8MinVal.u8Val, pstThrS16Ctrl->un8MidVal.u8Val,
                          pstThrS16Ctrl->un8MaxVal.u8Val, is_mmm);
  }
  CVI_IVE_BufFlush(pIveHandle, pstSrc);
  CVI_IVE_BufFlush(pIveHandle, pstDst);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Thresh_U16(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                           IVE_THRESH_U16_CTRL_S *pstThrU16Ctrl, bool bInstant) {
  if (!IsValidImageType(pstSrc, STRFY(pstSrc), IVE_IMAGE_TYPE_U16C1)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1)) {
    return CVI_FAILURE;
  }
  CVI_IVE_BufRequest(pIveHandle, pstSrc);
  CVI_IVE_BufRequest(pIveHandle, pstDst);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  u64 data_size = cpp_src->m_tg.stride.n / getFmtSize(cpp_src->m_tg.fmt);
  bool is_mmm = (pstThrU16Ctrl->enMode == IVE_THRESH_U16_MODE_U16_TO_U8_MIN_MID_MAX) ? true : false;
  neonU162U8ThresholdLH((u16 *)pstSrc->pu8VirAddr[0], (u8 *)pstDst->pu8VirAddr[0], data_size,
                        pstThrU16Ctrl->u16LowThr, pstThrU16Ctrl->u16HighThr,
                        pstThrU16Ctrl->u8MinVal, pstThrU16Ctrl->u8MidVal, pstThrU16Ctrl->u8MaxVal,
                        is_mmm);
  CVI_IVE_BufFlush(pIveHandle, pstSrc);
  CVI_IVE_BufFlush(pIveHandle, pstDst);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Xor(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  if (!IsValidImageType(pstSrc1, STRFY(pstSrc1), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstSrc2, STRFY(pstSrc2), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  if (!IsValidImageType(pstDst, STRFY(pstDst), IVE_IMAGE_TYPE_U8C1, IVE_IMAGE_TYPE_U8C3_PLANAR)) {
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_xor.init(&handle_ctx->ctx, handle_ctx->cvk_ctx);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs = {*cpp_dst};

  return handle_ctx->t_h.t_xor.run(&handle_ctx->ctx, handle_ctx->cvk_ctx, inputs, &outputs);
}

// ---------------------------------
// cpu functions
// ---------------------------------

/**
 * @Param Src gray image (size: wxh)
 * @Param Integral integral image (size: (w+1)x(h+1))
 * @Param Width image width
 * @Param Height image height
 * @Param Stride shift bytes
 */
inline void GetGrayIntegralImage(u8 *Src, u32 *Integral, int Width, int Height, int src_stride,
                                 int dst_stride) {
  memset(Integral, 0, dst_stride * sizeof(u32));

  for (int Y = 0; Y < Height; Y++) {
    u8 *LinePS = Src + Y * src_stride;
    u32 *LinePL = Integral + Y * (dst_stride) + 1;
    u32 *LinePD = Integral + (Y + 1) * (dst_stride) + 1;
    LinePD[-1] = 0;
    for (int X = 0, Sum = 0; X < Width; X++) {
      Sum += LinePS[X];
      LinePD[X] = LinePL[X] + Sum;
    }
  }
}

/**
 * @param cols image width
 * @param rows image column
 * @param image gray image
 * @param hist buffer for histogram values
 * @param num_bins how many values you want to find frequency
 */

inline int cal_hist(int cols, int rows, u8 *image, int src_stride, u32 *hist, int num_bins) {
  if (cols < 1 || rows < 1 || num_bins < 1) {
    return (1);
  }
  memset(hist, 0, sizeof(u32) * num_bins);

  for (int Y = 0; Y < rows; Y++) {
    u8 *LinePS = image + Y * src_stride;
    for (int X = 0, Sum = 0; X < cols; X++) {
      Sum = LinePS[X];
      hist[Sum]++;
    }
  }

  return (0);
}

/**
 * @param hist frequencies
 * @param eqhist new gray level (newly mapped pixel values)
 * @param nbr_elements total number of pixels
 * @param nbr_bins number of levels
 */

inline int equalize_hist(u32 *hist, u8 *eqhist, int nbr_elements, int nbr_bins) {
  int curr, i, total;
  if (nbr_elements < 1 || nbr_bins < 1) {
    return (1);
  }
  curr = 0;
  total = nbr_elements;
  // calculating cumulative frequency and new gray levels
  for (i = 0; i < nbr_bins; i++) {
    // cumulative frequency
    curr += hist[i];
    // calculating new gray level after multiplying by
    // maximum gray count which is 255 and dividing by
    // total number of pixels
    eqhist[i] = (u8)round((((float)curr) * 255) / total);
  }
  return (0);
}

inline int histogramEqualisation(int cols, int rows, u8 *image, int src_stride, u8 *pDst,
                                 int dst_stride) {
  u32 hist[256] = {0};
  u8 new_gray_level[256] = {0};
  int col, row, total, st;

  st = cal_hist(cols, rows, image, src_stride, hist, 256);
  if (st > 0) {
    return (st);
  }
  total = cols * rows;
  st = equalize_hist(hist, new_gray_level, total, 256);
  if (st > 0) {
    return (st);
  }
  u8 *ptr = image;
  for (row = 0; row < rows; row++) {
    for (col = 0; col < cols; col++) {
      pDst[col] = (unsigned char)new_gray_level[ptr[col]];
    }
    pDst += dst_stride;
    ptr += src_stride;
  }
  return st;
}

// main body
CVI_S32 CVI_IVE_Integ(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_MEM_INFO_S *pstDst,
                      IVE_INTEG_CTRL_S *ctrl, bool bInstant) {
  if (pstSrc->enType != IVE_IMAGE_TYPE_U8C1) {
    std::cerr << "Output only accepts U8C1 image format." << std::endl;
    return CVI_FAILURE;
  }

  CVI_IVE_BufRequest(pIveHandle, pstSrc);

  u32 *ptr = (u32 *)pstDst->pu8VirAddr;
  int channels = 1;
  int dst_stride = channels * (pstSrc->u16Width + 1);

  GetGrayIntegralImage((u8 *)pstSrc->pu8VirAddr[0], ptr, (int)pstSrc->u16Width,
                       (int)pstSrc->u16Height, (int)pstSrc->u16Stride[0], dst_stride);

  CVI_IVE_BufFlush(pIveHandle, pstSrc);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Hist(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_MEM_INFO_S *pstDst,
                     bool bInstant) {
  if (pstSrc->enType != IVE_IMAGE_TYPE_U8C1) {
    std::cerr << "Output only accepts U8C1 image format." << std::endl;
    return CVI_FAILURE;
  }

  cal_hist((int)pstSrc->u16Width, (int)pstSrc->u16Height, (u8 *)pstSrc->pu8VirAddr[0],
           (int)pstSrc->u16Stride[0], (u32 *)pstDst->pu8VirAddr, 256);
  CVI_IVE_BufRequest(pIveHandle, pstSrc);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_EqualizeHist(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc,
                             IVE_DST_IMAGE_S *pstDst, IVE_EQUALIZE_HIST_CTRL_S *ctrl,
                             bool bInstant) {
  if (pstSrc->enType != IVE_IMAGE_TYPE_U8C1) {
    std::cerr << "Output only accepts U8C1 image format." << std::endl;
    return CVI_FAILURE;
  }
  CVI_IVE_BufRequest(pIveHandle, pstSrc);

  histogramEqualisation((int)pstSrc->u16Width, (int)pstSrc->u16Height, (u8 *)pstSrc->pu8VirAddr[0],
                        (int)pstSrc->u16Stride[0], (u8 *)pstDst->pu8VirAddr[0],
                        (int)pstDst->u16Stride[0]);

  CVI_IVE_BufFlush(pIveHandle, pstSrc);

  return CVI_SUCCESS;
}

inline float cal_norm_cc(unsigned char *psrc1, unsigned char *psrc2, int srcw, int srch) {
  int i, wxh;
  uint t1, t2, t3;
  float rtv = 0;
  double d1, d2, d3;

  if (srcw < 1 || srch < 1) {
    return (0.0);
  }
  t1 = 0;
  t2 = 0;
  t3 = 0;
  wxh = srcw * srch;
  for (i = 0; i < wxh; i++) {
    t1 += (psrc1[i] * psrc2[i]);
  }

  for (i = 0; i < wxh; i++) {
    t2 += (psrc1[i] * psrc1[i]);
  }
  for (i = 0; i < wxh; i++) {
    t3 += (psrc2[i] * psrc2[i]);
  }
  if (t2 < 1 || t3 < 1) {
    return (0.0);
  }
  d1 = (double)(t1);
  d2 = sqrt((double)t2) * sqrt((double)t3);
  d3 = d1 / (d2 + 1);
  // printf("%lf %lf %d %d %d\n", d1, d2, t1, t2, t3);
  rtv = (float)(d3);

  return rtv;
}

CVI_S32 CVI_IVE_NCC(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_MEM_INFO_S *pstDst, bool bInstant) {
  if (pstSrc1->enType != IVE_IMAGE_TYPE_U8C1) {
    std::cerr << "Output only accepts U8C1 image 1 format." << std::endl;
    return CVI_FAILURE;
  }
  if (pstSrc2->enType != IVE_IMAGE_TYPE_U8C1) {
    std::cerr << "Output only accepts U8C1 image 2 format." << std::endl;
    return CVI_FAILURE;
  }

  CVI_IVE_BufRequest(pIveHandle, pstSrc1);
  CVI_IVE_BufRequest(pIveHandle, pstSrc2);
  float *ptr = (float *)pstDst->pu8VirAddr;
  float rt = cal_norm_cc((u8 *)pstSrc1->pu8VirAddr[0], (u8 *)pstSrc2->pu8VirAddr[0],
                         (int)pstSrc1->u16Width, (int)pstSrc1->u16Height);

  ptr[0] = rt;

  CVI_IVE_BufFlush(pIveHandle, pstSrc1);
  CVI_IVE_BufFlush(pIveHandle, pstSrc2);

  return CVI_SUCCESS;
}

inline void uint16_8bit(uint16_t *in, uint8_t *out, int dim) {
  int i;

  for (i = 0; i < dim; i++) {
    // uint16_t n = in[i];  // because shifting the sign bit invokes UB
    // uint8_t hi = ((n >> 8) & 0xff);
    uint8_t lo = ((in[i] >> 0) & 0xff);
    out[i] = lo;
  }
}

CVI_S32 CVI_IVE_16BitTo8Bit(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                            IVE_16BIT_TO_8BIT_CTRL_S *ctrl, bool bInstant) {
  if (pstSrc->enType != IVE_IMAGE_TYPE_U16C1) {
    std::cerr << "Input only accepts U16C1 image format." << std::endl;
    return CVI_FAILURE;
  }
  if (pstDst->enType != IVE_IMAGE_TYPE_U8C1) {
    std::cerr << "Output only accepts U8C1 image format." << std::endl;
    return CVI_FAILURE;
  }
  CVI_IVE_BufRequest(pIveHandle, pstSrc);
  CVI_IVE_BufRequest(pIveHandle, pstDst);

  int wxh = ((int)pstSrc->u16Width * (int)pstSrc->u16Height);
  uint16_8bit((uint16_t *)pstSrc->pu8VirAddr[0], (uint8_t *)pstDst->pu8VirAddr[0], wxh);

  CVI_IVE_BufFlush(pIveHandle, pstSrc);
  CVI_IVE_BufFlush(pIveHandle, pstDst);
  return CVI_SUCCESS;
}

typedef enum _LbpMappingType {
  LbpUniform /**< Uniform local binary patterns. */
} LbpMappingType;

/** @brief Local Binary Pattern extractor */
typedef struct cvLbp_ {
  u32 dimension;
  u32 mapping[256];
  bool transposed;
} cvLbp;

static void lbp_init_uniform(cvLbp *self) {
  int i, j;

  /* overall number of quantized LBPs */
  self->dimension = 58;

  /* all but selected patterns map to bin 57 (the first bin has index 0) */
  for (i = 0; i < 256; ++i) {
    self->mapping[i] = 57;
  }

  /* the uniform (all zeros or ones) patterns map to bin 56 */
  self->mapping[0x00] = 56;
  self->mapping[0xff] = 56;

  /* 56 uniform patterns */
  for (i = 0; i < 8; ++i) {
    for (j = 1; j <= 7; ++j) {
      int ip;
      int unsigned string;
      if (self->transposed) {
        ip = (-i + 2 - (j - 1) + 16) % 8;
      } else {
        ip = i;
      }

      /* string starting with j ones */
      string = (1 << j) - 1;
      string <<= ip;
      string = (string | (string >> 8)) & 0xff;

      self->mapping[string] = i * 7 + (j - 1);
    }
  }
}
u32 lbp_get_dimension(cvLbp *self) { return self->dimension; }

/** @brief Extract LBP features
 ** @param self LBP object.
 ** @param features buffer to write the features to.
 ** @param image image.
 ** @param width image width.
 ** @param height image height.
 ** @param cellSize size of the LBP cells. Note: 32x32
 **
 ** @a features is a  @c numColumns x @c numRows x @c dimension where
 ** @c dimension is the dimension of a LBP feature obtained from ::vl_lbp_get_dimension,
 ** @c numColumns is equal to @c floor(width / cellSize), and similarly
 ** for @c numRows.
 **/

void lbp_process(cvLbp *self, u8 *lbpimg, u8 *image, u32 stride, u32 width, u32 height) {
  // u32 cellSize = 32;
  // u32 cwidth = width / cellSize;
  // u32 cheight = height / cellSize;
  // u32 cstride = cwidth * cheight;
  // u32 cdimension = lbp_get_dimension(self);
  int x, y, bin;

#define at(u, v) (*(image + stride * (v) + (u)))
#define to(m, n) (*(lbpimg + stride * (n) + (m)))

  /* clear the output buffer */
  memset(lbpimg, 0, (stride * height));

  /* accumulate pixel-level measurements into cells */
  for (y = 1; y < (signed)height - 1; ++y) {
    // float wy1 = (y + 0.5f) / (float)cellSize - 0.5f;
    // int cy1 = (int)floor(wy1);
    // int cy2 = cy1 + 1;
    // float wy2 = wy1 - (float)cy1;
    // wy1 = 1.0f - wy2;
    // if (cy1 >= (signed)cheight) continue;

    for (x = 1; x < (signed)width - 1; ++x) {
      // float wx1 = (x + 0.5f) / (float)cellSize - 0.5f;
      // int cx1 = (int)floor(wx1);
      // int cx2 = cx1 + 1;
      // float wx2 = wx1 - (float)cx1;
      // wx1 = 1.0f - wx2;
      // if (cx1 >= (signed)cwidth) continue;

      {
        int unsigned bitString = 0;
        u8 center = at(x, y);
        if (at(x + 1, y + 0) > center) bitString |= 0x1 << 0; /*  E */
        if (at(x + 1, y + 1) > center) bitString |= 0x1 << 1; /* SE */
        if (at(x + 0, y + 1) > center) bitString |= 0x1 << 2; /* S  */
        if (at(x - 1, y + 1) > center) bitString |= 0x1 << 3; /* SW */
        if (at(x - 1, y + 0) > center) bitString |= 0x1 << 4; /*  W */
        if (at(x - 1, y - 1) > center) bitString |= 0x1 << 5; /* NW */
        if (at(x + 0, y - 1) > center) bitString |= 0x1 << 6; /* N  */
        if (at(x + 1, y - 1) > center) bitString |= 0x1 << 7; /* NE */
        bin = self->mapping[bitString];
        to(x, y) = bin;
      }

    } /* x */
  }   /* y */
}

CVI_S32 CVI_IVE_LBP(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                    IVE_LBP_CTRL_S *ctrl, bool bInstant) {
  if (pstSrc->enType != IVE_IMAGE_TYPE_U8C1) {
    std::cerr << "Input only accepts U8C1 image format." << std::endl;
    return CVI_FAILURE;
  }
  if (pstDst->enType != IVE_IMAGE_TYPE_U8C1) {
    std::cerr << "Output only accepts U8C1 image format." << std::endl;
    return CVI_FAILURE;
  }

  cvLbp self;
  lbp_init_uniform(&self);

  CVI_IVE_BufRequest(pIveHandle, pstSrc);

  lbp_process(&self, (u8 *)pstDst->pu8VirAddr[0], (u8 *)pstSrc->pu8VirAddr[0],
              (u32)pstSrc->u16Stride[0], (u32)pstSrc->u16Width, (int)pstSrc->u16Height);

  CVI_IVE_BufFlush(pIveHandle, pstSrc);

  return CVI_SUCCESS;
}

#include "avir/avir.h"

CVI_S32 CVI_IVE_Resize(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_RESIZE_CTRL_S *ctrl, bool bInstant) {
  if (pstSrc->enType != IVE_IMAGE_TYPE_U8C1) {
    std::cerr << "Input only accepts U8C1 image format." << std::endl;
    return CVI_FAILURE;
  }
  if (pstDst->enType != IVE_IMAGE_TYPE_U8C1) {
    std::cerr << "Output only accepts U8C1 image format." << std::endl;
    return CVI_FAILURE;
  }

  CVI_IVE_BufRequest(pIveHandle, pstSrc);

  avir::CImageResizer<> ImageResizer(8);
  ImageResizer.resizeImage((u8 *)pstSrc->pu8VirAddr[0], (int)pstSrc->u16Width,
                           (int)pstSrc->u16Height, 0, (u8 *)pstDst->pu8VirAddr[0],
                           (int)pstDst->u16Width, (int)pstDst->u16Height, 1, 0);

  CVI_IVE_BufFlush(pIveHandle, pstSrc);

  return CVI_SUCCESS;
}

#if 0
#include "cvi_vip.h"
CVI_S32 set_fmt_ex(CVI_S32 fd, CVI_S32 width, CVI_S32 height, CVI_U32 pxlfmt, CVI_U32 csc, CVI_U32 quant)
{
	struct v4l2_format fmt;
	//fmt.type = type;
	fmt.fmt.pix_mp.width = width;
	fmt.fmt.pix_mp.height = height;
	fmt.fmt.pix_mp.pixelformat = pxlfmt;
	fmt.fmt.pix_mp.field = V4L2_FIELD_ANY;
	if (pxlfmt == V4L2_PIX_CVK_FMT_RGBM){
	fmt.fmt.pix_mp.colorspace = V4L2_COLORSPACE_SRGB;
	}else{
	fmt.fmt.pix_mp.colorspace = csc;
	}

	fmt.fmt.pix_mp.quantization = quant;
	fmt.fmt.pix_mp.num_planes = 3;

	if (-1 == ioctl(fd, VIDIOC_TRY_FMT, &fmt)){
		perror("VIDIOC_TRY_FMT");
		//printf("VIDIOC_TRY_FMT");
	}
	if (-1 == ioctl(fd, VIDIOC_S_FMT, &fmt)){
		perror("VIDIOC_S_FMT");
		//printf("VIDIOC_S_FMT");
	}
	return(fmt.fmt.pix.sizeimage);
}


CVI_S32 CVI_IVE_CSC(IVE_HANDLE *pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst
			, IVE_CSC_CTRL_S *pstCscCtrl, CVI_BOOL bInstant)
{
	CVI_U32 srcfmt, dstfmt, srcCSC, dstCSC, srcQuant, dstQuant;
	struct vdev *srcd, *dstd;
	struct buffer buf;
	CVI_S32 ret;
#if 0
	srcd = get_dev_info(VDEV_TYPE_IMG, 0);
	dstd = get_dev_info(VDEV_TYPE_SC, 0);

  srcfmt = V4L2_PIX_CVK_FMT_YUV420M;
  dstfmt = V4L2_PIX_CVK_FMT_RGBM;
	srcCSC = V4L2_COLORSPACE_REC709;
	srcQuant = V4L2_QUANTIZATION_DEFAULT;
	dstCSC = V4L2_COLORSPACE_REC709;
	dstQuant = V4L2_QUANTIZATION_LIM_RANGE;
#endif

#if 0
	switch (pstCscCtrl->enMode) {
  case IVE_CSC_MODE_PIC_BT601_YUV2HSV:
	case IVE_CSC_MODE_VIDEO_BT601_YUV2RGB:
	case IVE_CSC_MODE_PIC_BT601_YUV2RGB:
  case IVE_CSC_MODE_VIDEO_BT709_YUV2RGB:
  case IVE_CSC_MODE_PIC_BT601_YUV2HSV:
	case IVE_CSC_MODE_PIC_BT601_YUV2HSV:
		if (srcfmt == V4L2_PIX_CVK_FMT_RGBM) {
			//CVI_TRACE(CVI_DBG_ERR, CVI_ID_IVE, "Invalid parameters\n");
			return CVI_FAILURE;
		}
		srcCSC = V4L2_COLORSPACE_SMPTE170M;
		srcQuant = V4L2_QUANTIZATION_DEFAULT;
		break;
	case IVE_CSC_MODE_VIDEO_BT709_YUV2RGB:
	case IVE_CSC_MODE_PIC_BT709_YUV2RGB:
	case IVE_CSC_MODE_PIC_BT709_YUV2HSV:
		if (srcfmt == V4L2_PIX_CVK_FMT_RGBM) {
			//CVI_TRACE(CVI_DBG_ERR, CVI_ID_IVE, "Invalid parameters\n");
			return CVI_FAILURE;
		}
		srcCSC = V4L2_COLORSPACE_REC709;
		srcQuant = V4L2_QUANTIZATION_DEFAULT;
		break;
	case IVE_CSC_MODE_VIDEO_BT601_RGB2YUV:
	case IVE_CSC_MODE_VIDEO_BT709_RGB2YUV:
	case IVE_CSC_MODE_PIC_BT601_RGB2YUV:
	case IVE_CSC_MODE_PIC_BT709_RGB2YUV:
		if (srcfmt != V4L2_PIX_CVK_FMT_RGBM) {
			//CVI_TRACE(CVI_DBG_ERR, CVI_ID_IVE, "Invalid parameters\n");
			return CVI_FAILURE;
		}
		srcCSC = V4L2_COLORSPACE_SRGB;
		srcQuant = V4L2_QUANTIZATION_DEFAULT;
		break;
	}
#endif

	//srcCSC = V4L2_COLORSPACE_REC709;
	//srcQuant = V4L2_QUANTIZATION_DEFAULT;

#if 0
	set_fmt_ex(srcd->fd, pstSrc->u16Width, pstSrc->u16Height, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
		, srcfmt, srcCSC, srcQuant);
	set_fmt_ex(dstd->fd, pstDst->u16Width, pstDst->u16Height, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
		, dstfmt, dstCSC, dstQuant);

	for (CVI_U8 i = 0; i < 3; ++i) {
		buf.phy_addr[i] = pstSrc->u64PhyAddr[i];
		buf.length[i] = pstSrc->u16Stride[i] * pstSrc->u16Height;
		if (pstSrc->enType == IVE_IMAGE_TYPE_YUV420P && i > 0)
			buf.length[i] >>= 1;
	}
	qbuf(srcd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, &buf);
	for (CVI_U8 i = 0; i < 3; ++i) {
		buf.phy_addr[i] = pstDst->u64PhyAddr[i];
		buf.length[i] = pstDst->u16Stride[i] * pstDst->u16Height;
		if (pstDst->enType == IVE_IMAGE_TYPE_YUV420P && i > 0)
			buf.length[i] >>= 1;
	}
	qbuf(dstd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, &buf);


	do {
		fd_set wfds;
		struct timeval tv;

		FD_ZERO(&wfds);
		FD_SET(dstd->fd, &wfds);
		tv.tv_sec = 0;
		tv.tv_usec = 500 * 1000;
		ret = select(dstd->fd + 1, NULL, &wfds, NULL, &tv);
		if (ret == -1) {
			if (errno == EINTR)
				continue;
			//CVI_TRACE(CVI_DBG_ERR, CVI_ID_IVE, "select error\n");
			break;
		}

		if (ret == 0) {
			//CVI_TRACE(CVI_DBG_ERR, CVI_ID_IVE, "select timeout\n");
			ret = CVI_FAILURE;
			break;
		}

		if (FD_ISSET(dstd->fd, &wfds)) {
			ret = CVI_SUCCESS;
			break;
		}
	} while(1);
#endif
	return ret;
}

#else

float max(float a, float b, float c) { return ((a > b) ? (a > c ? a : c) : (b > c ? b : c)); }
float min(float a, float b, float c) { return ((a < b) ? (a < c ? a : c) : (b < c ? b : c)); }
int rgb_to_hsv(float r, float g, float b, float &h, float &s, float &v) {
  // R, G, B values are divided by 255
  // to change the range from 0..255 to 0..1:
  // float h, s, v;
  h = 0;
  s = 0;
  v = 0;

  r /= 255.0;
  g /= 255.0;
  b /= 255.0;
  float cmax = max(r, g, b);       // maximum of r, g, b
  float cmin = min(r, g, b);       // minimum of r, g, b
  float diff = cmax - cmin + 1.0;  // diff of cmax and cmin.
  if (cmax == cmin)
    h = 0;
  else if (cmax == r)
    h = fmod((60 * ((g - b) / diff) + 360), 360.0);
  else if (cmax == g)
    h = fmod((60 * ((b - r) / diff) + 120), 360.0);
  else if (cmax == b)
    h = fmod((60 * ((r - g) / diff) + 240), 360.0);
  // if cmax equal zero
  if (cmax == 0) {
    s = 0;
  } else {
    s = (diff / cmax) * 100;
  }
  // compute v
  v = cmax * 100;
  // printf("h s v=(%f, %f, %f)\n", h, s, v );
  return 0;
}

int rgbToHsv(unsigned char *rgb, unsigned char *hsv, int srcw, int srch, int ch) {
  int i;
  int wxh = srcw * srch;
  float r, g, b, h, s, v;

  for (i = 0; i < wxh; i++) {
    r = rgb[ch * i] / 255.0;
    g = rgb[ch * i + 1] / 255.0;
    b = rgb[ch * i + 2] / 255.0;
    rgb_to_hsv(r, g, b, h, s, v);

    hsv[ch * i] = floor(h);
    hsv[ch * i + 1] = floor(s);
    hsv[ch * i + 2] = floor(v);
  }

  return 0;
}

void rgbToGray(unsigned char *rgb, unsigned char *gray, int stride, int srcw, int srch) {
  int i, j, n, ii;
  uint r, g, b;
  unsigned char one_gray;
  unsigned char *pr, *pg, *pb;
  pr = rgb;
  pg = rgb + srcw * srch;
  pb = rgb + srcw * srch * 2;

  for (i = 0; i < srch; i++) {
    n = 0;
    ii = i * srcw;
    for (j = 0; j < srcw; j++, n++) {
      r = *(pr + ii + j);          // red
      g = *(pg + ii + j);          // green
      b = *(pb + ii + j);          // blue
      one_gray = (r + g + b) / 3;  //(r*19595 + g*38469 + b*7472) >> 16;
      *(gray + ii + n) = one_gray;
    }
  }
}

CVI_S32 CVI_IVE_CSC(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                    IVE_CSC_CTRL_S *ctrl, bool bInstant) {
  if (pstSrc->enType != IVE_IMAGE_TYPE_U8C3_PLANAR) {
    std::cerr << "Input only accepts U8C3_PLANAR image format." << std::endl;
    return CVI_FAILURE;
  }

  CVI_IVE_BufRequest(pIveHandle, pstSrc);
  int strideSrc;  //, strideDst;

  switch (ctrl->enMode) {
    case IVE_CSC_MODE_PIC_RGB2HSV:
      rgbToHsv((u8 *)pstSrc->pu8VirAddr[0], (u8 *)pstDst->pu8VirAddr[0], (int)pstSrc->u16Width,
               (int)pstSrc->u16Height, 3);
      break;

    case IVE_CSC_MODE_PIC_RGB2GRAY:
      strideSrc = pstSrc->u16Stride[0];
      // strideDst = pstDst->u16Stride[0];
      // printf("strideSrc: %d, strideDat: %d\n", strideSrc, strideDst);
      rgbToGray((u8 *)pstSrc->pu8VirAddr[0], (u8 *)pstDst->pu8VirAddr[0], strideSrc,
                (int)pstSrc->u16Width, (int)pstSrc->u16Height);
      break;

    default:
      strideSrc = pstSrc->u16Stride[0];
      // strideDst = pstDst->u16Stride[0];
      // printf("strideSrc: %d, strideDat: %d\n", strideSrc, strideDst);
      rgbToGray((u8 *)pstSrc->pu8VirAddr[0], (u8 *)pstDst->pu8VirAddr[0], strideSrc,
                (int)pstSrc->u16Width, (int)pstSrc->u16Height);

      break;
  }

  CVI_IVE_BufFlush(pIveHandle, pstSrc);

  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_FilterAndCSC(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc,
                             IVE_SRC_IMAGE_S *pstBuf, IVE_DST_IMAGE_S *pstDst,
                             IVE_FILTER_AND_CSC_CTRL_S *ctrl, bool bInstant) {
  if (pstBuf->enType != IVE_IMAGE_TYPE_U8C3_PLANAR) {
    std::cerr << "Input only accepts U8C3_PLANAR image format." << std::endl;
    return CVI_FAILURE;
  }

  IVE_FILTER_CTRL_S iveFltCtrl;
  iveFltCtrl.u8MaskSize = 5;
  memcpy(iveFltCtrl.as8Mask, ctrl->as8Mask, 25 * sizeof(CVI_S8));
  iveFltCtrl.u32Norm = 273;
  CVI_IVE_Filter(pIveHandle, pstSrc, pstBuf, &iveFltCtrl, 0);

  memcpy(pstBuf->pu8VirAddr[0], pstSrc->pu8VirAddr[0], pstBuf->u16Stride[0] * pstBuf->u16Height);
  memcpy(pstBuf->pu8VirAddr[1], pstSrc->pu8VirAddr[0], pstBuf->u16Stride[0] * pstBuf->u16Height);
  memcpy(pstBuf->pu8VirAddr[2], pstSrc->pu8VirAddr[0], pstBuf->u16Stride[0] * pstBuf->u16Height);

  CVI_IVE_BufRequest(pIveHandle, pstBuf);
  int strideSrc;  //, strideDst;

  switch (ctrl->enMode) {
    case IVE_CSC_MODE_PIC_RGB2HSV:
      rgbToHsv((u8 *)pstBuf->pu8VirAddr[0], (u8 *)pstDst->pu8VirAddr[0], (int)pstBuf->u16Width,
               (int)pstBuf->u16Height, 3);
      break;

    case IVE_CSC_MODE_PIC_RGB2GRAY:
      strideSrc = pstBuf->u16Stride[0];
      // strideDst = pstDst->u16Stride[0];
      // printf("strideSrc: %d, strideDat: %d\n", strideSrc, strideDst);
      rgbToGray((u8 *)pstBuf->pu8VirAddr[0], (u8 *)pstDst->pu8VirAddr[0], strideSrc,
                (int)pstBuf->u16Width, (int)pstBuf->u16Height);
      break;

    default:
      strideSrc = pstBuf->u16Stride[0];
      // strideDst = pstDst->u16Stride[0];
      // printf("strideSrc: %d, strideDat: %d\n", strideSrc, strideDst);
      rgbToGray((u8 *)pstBuf->pu8VirAddr[0], (u8 *)pstDst->pu8VirAddr[0], strideSrc,
                (int)pstBuf->u16Width, (int)pstBuf->u16Height);

      break;
  }

  CVI_IVE_BufFlush(pIveHandle, pstBuf);

  return CVI_SUCCESS;
}

#endif