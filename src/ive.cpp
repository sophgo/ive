#include "ive.h"
#include <glog/logging.h>
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
#include "tpu/tpu_copy.hpp"
#include "tpu/tpu_filter.hpp"
#include "tpu/tpu_magandang.hpp"
#include "tpu/tpu_morph.hpp"
#include "tpu/tpu_normalize.hpp"
#include "tpu/tpu_or.hpp"
#include "tpu/tpu_sad.hpp"
#include "tpu/tpu_sigmoid.hpp"
#include "tpu/tpu_sobel.hpp"
#include "tpu/tpu_sub.hpp"
#include "tpu/tpu_table.hpp"
#include "tpu/tpu_threshold.hpp"
#include "tpu/tpu_xor.hpp"

#include <cmath>
#include <limits>

struct TPU_HANDLE {
  TblMgr t_tblmgr;
  IveTPUAdd t_add;
  IveTPUAnd t_and;
  IveTPUBlock t_block;
  IveTPUBlockBF16 t_block_bf16;
  IveTPUCopyInterval t_copy_int;
  IveTPUErode t_erode;
  IveTPUFilter t_filter;
  IveTPUFilterBF16 t_filter_bf16;
  IveTPUMagAndAng t_magandang;
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
  bmk1880v2_context_t *bk_ctx;
  TPU_HANDLE t_h;
  // VIP
};

void CVI_SYS_LOGGING(char *argv0) { google::InitGoogleLogging(argv0); }

IVE_HANDLE CVI_IVE_CreateHandle() {
  IVE_HANDLE_CTX *handle_ctx = new IVE_HANDLE_CTX;
  createHandle(&handle_ctx->ctx, &handle_ctx->bk_ctx);
  handle_ctx->t_h.t_tblmgr.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  return (void *)handle_ctx;
}

CVI_S32 CVI_IVE_DestroyHandle(IVE_HANDLE pIveHandle) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_tblmgr.free(&handle_ctx->ctx);
  destroyHandle(&handle_ctx->ctx);
  delete handle_ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_BufFlush(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  // FIXME: Hack for U32 images.
  if (pstImg->tpu_block != NULL) {
    auto *img = reinterpret_cast<CviImg *>(pstImg->tpu_block);
    if (img->Flush(&handle_ctx->ctx) != BM_SUCCESS) {
      return CVI_FAILURE;
    }
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_BufRequest(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  // FIXME: Hack for U32 images.
  if (pstImg->tpu_block != NULL) {
    auto *img = reinterpret_cast<CviImg *>(pstImg->tpu_block);
    if (img->Invld(&handle_ctx->ctx) != BM_SUCCESS) {
      return CVI_FAILURE;
    }
  }
  return CVI_SUCCESS;
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

CVI_S32 CVI_IVE_CreateImage(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg, IVE_IMAGE_TYPE_E enType,
                            u16 u16Width, u16 u16Height) {
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
  int c = 1;
  int fmt_size = 1;
  fmt_t fmt = FMT_U8;
  switch (enType) {
    case IVE_IMAGE_TYPE_S8C1:
      fmt = FMT_I8;
      break;
    case IVE_IMAGE_TYPE_U8C1:
      break;
    case IVE_IMAGE_TYPE_U8C3_PACKAGE:
    case IVE_IMAGE_TYPE_U8C3_PLANAR:
      c = 3;
      break;
    case IVE_IMAGE_TYPE_BF16C1:
      fmt_size = 2;
      fmt = FMT_BF16;
      break;
    case IVE_IMAGE_TYPE_U16C1:
      fmt_size = 2;
      fmt = FMT_U16;
      break;
    case IVE_IMAGE_TYPE_S16C1:
      fmt_size = 2;
      fmt = FMT_I16;
      break;
    case IVE_IMAGE_TYPE_U32C1:
      fmt_size = 4;
      fmt = FMT_U32;
      break;
    case IVE_IMAGE_TYPE_FP32C1:
      fmt_size = 4;
      fmt = FMT_F32;
      break;
    default:
      std::cerr << "Not supported enType " << enType << std::endl;
      return CVI_FAILURE;
      break;
  }
  int total_size = c * u16Width * u16Height * fmt_size;
  // Special case for unsupported I32/U32 images
  // FIXME: Put thosinto bmkernel, bmruntime
  if (fmt == FMT_U32) {
    pstImg->tpu_block = NULL;
    pstImg->enType = enType;
    pstImg->u16Width = u16Width;
    pstImg->u16Height = u16Height;
    pstImg->u16Reserved = fmt_size;
    int img_sz = pstImg->u16Width * pstImg->u16Height * fmt_size * c;
    uint8_t *arr = new uint8_t[img_sz];
    for (size_t i = 0; i < (size_t)c; i++) {
      pstImg->pu8VirAddr[i] = arr + i * img_sz;
      pstImg->u64PhyAddr[i] = i * img_sz;
      pstImg->u16Stride[i] = pstImg->u16Width;
    }

    for (size_t i = c; i < 3; i++) {
      pstImg->pu8VirAddr[i] = NULL;
      pstImg->u64PhyAddr[i] = -1;
      pstImg->u16Stride[i] = 0;
    }
    return CVI_SUCCESS;
  }

  auto *cpp_img = new CviImg(&handle_ctx->ctx, c, u16Height, u16Width, fmt);
  pstImg->tpu_block = reinterpret_cast<CVI_IMG *>(cpp_img);

  pstImg->enType = enType;
  pstImg->u16Width = cpp_img->m_tg.shape.w;
  pstImg->u16Height = cpp_img->m_tg.shape.h;
  pstImg->u16Reserved = fmt_size;

  int img_sz = cpp_img->m_tg.stride.h * pstImg->u16Height * fmt_size;
  for (size_t i = 0; i < cpp_img->m_tg.shape.c; i++) {
    pstImg->pu8VirAddr[i] = cpp_img->GetVAddr() + i * img_sz;
    pstImg->u64PhyAddr[i] = cpp_img->GetPAddr() + i * img_sz;
    pstImg->u16Stride[i] = cpp_img->m_tg.stride.h / fmt_size;
  }

  for (size_t i = cpp_img->m_tg.shape.c; i < 3; i++) {
    pstImg->pu8VirAddr[i] = NULL;
    pstImg->u64PhyAddr[i] = -1;
    pstImg->u16Stride[i] = 0;
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_SubImage(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                         CVI_U16 u16X1, CVI_U16 u16Y1, CVI_U16 u16X2, CVI_U16 u16Y2) {
  if (pstSrc->tpu_block == NULL) {
    std::cerr << "Currently not support I32/ U32 sub image." << std::endl;
    return CVI_FAILURE;
  }
  if (u16X1 >= u16X2 || u16Y1 >= u16Y2) {
    std::cerr << "(X1, Y1) must smaller than (X2, Y2)." << std::endl;
    return CVI_FAILURE;
  }
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  auto *src_img = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  auto *cpp_img = new CviImg(&handle_ctx->ctx, *src_img, u16X1, u16Y1, u16X2, u16Y2);
  if (cpp_img == nullptr) {
    return CVI_FAILURE;
  }
  pstDst->tpu_block = reinterpret_cast<CVI_IMG *>(cpp_img);

  pstDst->enType = pstSrc->enType;
  pstDst->u16Width = cpp_img->m_tg.shape.w;
  pstDst->u16Height = cpp_img->m_tg.shape.h;
  pstDst->u16Reserved = pstSrc->u16Reserved;

  int img_sz = cpp_img->m_tg.stride.h * pstSrc->u16Height * pstSrc->u16Reserved;
  for (size_t i = 0; i < cpp_img->m_tg.shape.c; i++) {
    pstDst->pu8VirAddr[i] = cpp_img->GetVAddr() + i * img_sz;
    pstDst->u64PhyAddr[i] = cpp_img->GetPAddr() + i * img_sz;
    pstDst->u16Stride[i] = cpp_img->m_tg.stride.h;
  }

  for (size_t i = cpp_img->m_tg.shape.c; i < 3; i++) {
    pstDst->pu8VirAddr[i] = NULL;
    pstDst->u64PhyAddr[i] = -1;
    pstDst->u16Stride[i] = 0;
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
    default:
      std::cerr << "Not support channel " << enType;
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
    memcpy(img.pu8VirAddr[0], stbi_data, desiredNChannels * width * height);
    CVI_IVE_BufFlush(pIveHandle, &img);
    stbi_image_free(stbi_data);
  }
  return img;
}

CVI_S32 CVI_IVE_WriteImage(IVE_HANDLE pIveHandle, const char *filename, IVE_IMAGE_S *pstImg) {
  int desiredNChannels = -1;
  switch (pstImg->enType) {
    case IVE_IMAGE_TYPE_U8C1:
      desiredNChannels = STBI_grey;
      break;
    case IVE_IMAGE_TYPE_U8C3_PLANAR:
      desiredNChannels = STBI_rgb;
      break;
    default:
      std::cerr << "Not support channel " << pstImg->enType;
      return CVI_FAILURE;
      break;
  }
  CVI_IVE_BufRequest(pIveHandle, pstImg);
  stbi_write_png(filename, pstImg->u16Width, pstImg->u16Height, desiredNChannels,
                 pstImg->pu8VirAddr[0], pstImg->u16Stride[0]);
  return CVI_SUCCESS;
}

CVI_S32 CVI_SYS_FreeM(IVE_HANDLE pIveHandle, IVE_MEM_INFO_S *pstMemInfo) {
  delete[] pstMemInfo->pu8VirAddr;
  pstMemInfo->u32ByteSize = 0;
  return CVI_SUCCESS;
}

CVI_S32 CVI_SYS_FreeI(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  if (pstImg->tpu_block == NULL) {
    delete[] pstImg->pu8VirAddr[0];
  } else {
    auto *cpp_img = reinterpret_cast<CviImg *>(pstImg->tpu_block);
    cpp_img->Free(&handle_ctx->ctx);
    delete cpp_img;
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_DMA(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                    IVE_DMA_CTRL_S *pstDmaCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  int ret = CVI_FAILURE;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  if (pstDmaCtrl->enMode == IVE_DMA_MODE_DIRECT_COPY) {
    ret = CVI_SUCCESS;
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

    IveTPUCopyDirect::run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
#endif
  } else if (pstDmaCtrl->enMode == IVE_DMA_MODE_INTERVAL_COPY) {
    handle_ctx->t_h.t_copy_int.setInvertal(pstDmaCtrl->u8HorSegSize, pstDmaCtrl->u8VerSegRows);
    handle_ctx->t_h.t_copy_int.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
    CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
    std::vector<CviImg> inputs = {*cpp_src};
    std::vector<CviImg> outputs = {*cpp_dst};

    handle_ctx->t_h.t_copy_int.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs, true);
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
    if (cpp_src->m_tg.fmt == FMT_BF16 && cpp_dst->m_tg.fmt == FMT_F32) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      float *dst_ptr = (float *)cpp_dst->GetVAddr();
      u64 img_size = cpp_src->GetImgSize() / 2;
      neonBF162F32(src_ptr, dst_ptr, img_size);
      cpp_src->Flush(&handle_ctx->ctx);
      cpp_dst->Flush(&handle_ctx->ctx);
    } else if (cpp_src->m_tg.fmt == FMT_BF16 && cpp_dst->m_tg.fmt == FMT_U16) {
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
    } else if (cpp_src->m_tg.fmt == FMT_U16 &&
               (cpp_dst->m_tg.fmt == FMT_U8 || cpp_dst->m_tg.fmt == FMT_I8)) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      u8 *dst_ptr = (u8 *)cpp_dst->GetVAddr();
      u16 min = 65535, max = 0;
      u64 img_size = cpp_src->m_tg.shape.c * cpp_src->m_tg.shape.h * cpp_src->m_tg.shape.w;
      neonU16FindMinMax(src_ptr, img_size, &min, &max);
      if (cpp_dst->m_tg.fmt == FMT_U8) {
        neonU162U8Normalize(src_ptr, dst_ptr, img_size, min, max);
      } else {
        neonU162S8Normalize(src_ptr, (s8 *)dst_ptr, img_size, min, max);
      }
      cpp_src->Flush(&handle_ctx->ctx);
      cpp_dst->Flush(&handle_ctx->ctx);
    } else if (cpp_src->m_tg.fmt == FMT_BF16 &&
               (cpp_dst->m_tg.fmt == FMT_U8 || cpp_dst->m_tg.fmt == FMT_I8)) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      u8 *dst_ptr = (u8 *)cpp_dst->GetVAddr();
      float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
      u64 img_size = cpp_src->m_tg.shape.c * cpp_src->m_tg.shape.h * cpp_src->m_tg.shape.w;
      neonBF16FindMinMax(src_ptr, img_size, &min, &max);
      handle_ctx->t_h.t_norm.setMinMax(min, max);
      handle_ctx->t_h.t_norm.setOutputFMT(cpp_dst->m_tg.fmt);
      handle_ctx->t_h.t_norm.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
      CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
      CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
      std::vector<CviImg> inputs = {*cpp_src};
      std::vector<CviImg> outputs = {*cpp_dst};
      handle_ctx->t_h.t_norm.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
    }
  } else if (pstItcCtrl->enType == IVE_ITC_SATURATE) {
    if (cpp_src->m_tg.fmt == FMT_BF16 && cpp_dst->m_tg.fmt == FMT_F32) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      float *dst_ptr = (float *)cpp_dst->GetVAddr();
      u64 img_size = cpp_src->GetImgSize() / 2;
      neonBF162F32(src_ptr, dst_ptr, img_size);
      cpp_src->Flush(&handle_ctx->ctx);
      cpp_dst->Flush(&handle_ctx->ctx);
    } else if (cpp_src->m_tg.fmt == FMT_BF16 && cpp_dst->m_tg.fmt == FMT_U16) {
      cpp_src->Invld(&handle_ctx->ctx);
      cpp_dst->Invld(&handle_ctx->ctx);
      u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
      u16 *dst_ptr = (u16 *)cpp_dst->GetVAddr();
      u64 img_size = cpp_src->GetImgSize() / 2;
      neonBF162U16(src_ptr, dst_ptr, img_size);
      cpp_src->Flush(&handle_ctx->ctx);
      cpp_dst->Flush(&handle_ctx->ctx);
    } else if ((cpp_src->m_tg.fmt == FMT_BF16 || cpp_src->m_tg.fmt == FMT_U8 ||
                cpp_src->m_tg.fmt == FMT_I8) &&
               (cpp_dst->m_tg.fmt == FMT_BF16 || cpp_dst->m_tg.fmt == FMT_U8 ||
                cpp_dst->m_tg.fmt == FMT_I8)) {
      CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
      CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
      std::vector<CviImg> inputs = {*cpp_src};
      std::vector<CviImg> outputs = {*cpp_dst};

      IveTPUCopyDirect::run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
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
  int ret = CVI_FAILURE;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);

  const float &x = ctrl->aX;
  const float &y = ctrl->bY;
  if ((x == 1 && y == 1) || (x == 0.f && y == 0.f)) {
    ret = CVI_SUCCESS;
    handle_ctx->t_h.t_add.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
    CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
    CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
    std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
    std::vector<CviImg> outputs = {*cpp_dst};

    handle_ctx->t_h.t_add.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  }
  return ret;
}

CVI_S32 CVI_IVE_And(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_and.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs = {*cpp_dst};

  handle_ctx->t_h.t_and.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_BLOCK(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                      IVE_BLOCK_CTRL_S *pstBlkCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
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
  if ((cpp_src->m_tg.fmt != FMT_U8 && cpp_src->m_tg.fmt != FMT_BF16) &&
      (cpp_dst->m_tg.fmt != FMT_U8 && cpp_dst->m_tg.fmt != FMT_BF16)) {
    std::cerr << "CVI Block only supports U8/ BF16." << std::endl;
    return CVI_FAILURE;
  }
  if (cpp_src->m_tg.fmt == FMT_U8 && cpp_dst->m_tg.fmt == FMT_U8) {
    handle_ctx->t_h.t_block.setBinNum(pstBlkCtrl->f32BinSize);
    handle_ctx->t_h.t_block.setCellSize(u32CellSize, cpp_src->m_tg.shape.c);
    handle_ctx->t_h.t_block.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_block.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs, true);
  } else {
    handle_ctx->t_h.t_block_bf16.setBinNum(pstBlkCtrl->f32BinSize);
    handle_ctx->t_h.t_block_bf16.setCellSize(u32CellSize, cpp_src->m_tg.shape.c);
    handle_ctx->t_h.t_block_bf16.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_block_bf16.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs, true);
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Dilate(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_DILATE_CTRL_S *pstDilateCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_filter.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};

  u32 npu_num = handle_ctx->t_h.t_erode.getNpuNum();
  CviImg cimg(&handle_ctx->ctx, npu_num, 5, 5, FMT_U8);
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
  handle_ctx->t_h.t_filter.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  kernel.img.Free(&handle_ctx->ctx);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Erode(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                      IVE_ERODE_CTRL_S *pstErodeCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_erode.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};

  u32 npu_num = handle_ctx->t_h.t_erode.getNpuNum();
  CviImg cimg(&handle_ctx->ctx, npu_num, 5, 5, FMT_U8);
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
  handle_ctx->t_h.t_erode.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  kernel.img.Free(&handle_ctx->ctx);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Filter(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_FILTER_CTRL_S *pstFltCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_filter.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};

  if (pstFltCtrl->u8MaskSize != 3 && pstFltCtrl->u8MaskSize != 5) {
    std::cerr << "Currently Filter only supports filter size 3 or 5." << std::endl;
  }
  u32 npu_num = handle_ctx->t_h.t_filter.getNpuNum();
  CviImg cimg(&handle_ctx->ctx, npu_num, pstFltCtrl->u8MaskSize, pstFltCtrl->u8MaskSize, FMT_I8);
  IveKernel kernel;
  kernel.img = cimg;
  kernel.img.GetVAddr();
  int mask_length = pstFltCtrl->u8MaskSize * pstFltCtrl->u8MaskSize;
  for (size_t i = 0; i < npu_num; i++) {
    memcpy(kernel.img.GetVAddr() + i * mask_length, pstFltCtrl->as8Mask, mask_length);
  }
  kernel.img.Flush(&handle_ctx->ctx);
  kernel.multiplier.f = 1.f / pstFltCtrl->u32Norm;
  QuantizeMultiplierSmallerThanOne(kernel.multiplier.f, &kernel.multiplier.base,
                                   &kernel.multiplier.shift);
  handle_ctx->t_h.t_filter.setKernel(kernel);
  handle_ctx->t_h.t_filter.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  kernel.img.Free(&handle_ctx->ctx);
  return CVI_SUCCESS;
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
                             CVI_U16 u16CellSize, CVI_U16 u16BlkSize, CVI_U16 u16BlkStepX,
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
  if (!get_hog_feature_info(u16Width, u16Height, u16CellSize, u16BlkSize, &width_cell, &height_cell,
                            &width_block, &height_block)) {
    std::cerr << "Block size exceed cell block." << std::endl;
    return CVI_FAILURE;
  }
  u32 block_length = u16BlkSize * u16BlkSize;
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
                            pstHogCtrl->u16BlkSize, &width_cell, &height_cell, &width_block,
                            &height_block)) {
    std::cerr << "Block size exceed cell block." << std::endl;
    return CVI_FAILURE;
  }
  u32 &&cell_length = pstHogCtrl->u32CellSize * pstHogCtrl->u32CellSize;
  u32 &&cell_hist_length = height_cell * width_cell * pstHogCtrl->u8BinSize;
  u32 &&block_length = pstHogCtrl->u16BlkSize * pstHogCtrl->u16BlkSize;
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
        cell_histogram[cell_index + bin_index] += mag;
      } else {
        u32 bin_index = bin_div;
        u32 bin_index_2 = (bin_index + 1);
        if (bin_index_2 >= pstHogCtrl->u8BinSize) bin_index_2 = 0;
        float bin_div_dec_left = 1.f - bin_div_dec;
        cell_histogram[cell_index + bin_index] += (mag * bin_div_dec_left);
        cell_histogram[cell_index + bin_index_2] += (mag * bin_div_dec);
      }
    }
  }
  // for (u32 i = 1; i < (u32)(pstDstAng->u16Height - 1); i++) {
  //   u32 &&row_skip = pstDstAng->u16Stride[0] * i;
  //   for (u32 j = 1; j < (u32)(pstDstAng->u16Width - 1); j++) {
  //     float degree = convert_bf16_fp32(cell_ptr[j + row_skip]);
  //     u32 bin_index = degree < 0 ? (u32)((360.f + degree) / div) : (u32)(degree / div);
  //     u32 &&cell_index =
  //         (u32)((i / pstHogCtrl->u32CellSize) * width_cell + (u32)(j / pstHogCtrl->u32CellSize))
  //         * pstHogCtrl->u8BinSize;
  //     if (bin_index >= pstHogCtrl->u8BinSize) {
  //       std::cerr << "Pixel value " << degree << " at " << i << ", " << j << " exceed bin size "
  //                 << pstHogCtrl->u8BinSize << std::endl;
  //       Tracer::TraceEnd();
  //       Tracer::TraceEnd();
  //       return CVI_FAILURE;
  //     }
  //     cell_histogram[cell_index + bin_index]++;
  //   }
  // }
  Tracer::TraceEnd();

  Tracer::TraceBegin("Generate HOG histogram");
  u32 &&copy_length = pstHogCtrl->u8BinSize * pstHogCtrl->u16BlkSize;
  u32 &&copy_data_length = copy_length * sizeof(float);
  float *hog_ptr = (float *)pstDstHist->pu8VirAddr;
  memset(hog_ptr, 0, pstDstHist->u32ByteSize);
  u32 count = 0;
  for (u32 i = 0; i < height_block; i += pstHogCtrl->u16BlkStepY) {
    u32 &&row_skip = i * width_block;
    for (u32 j = 0; j < width_block; j += pstHogCtrl->u16BlkStepX) {
      u32 &&skip = j + row_skip;
      for (u32 k = 0; k < pstHogCtrl->u16BlkSize; k++) {
        u32 &&index = skip + k * width_block;
        auto *cell_hist_ptr = cell_histogram + index;
        auto *dst_hog_ptr = hog_ptr + count;
        memcpy(dst_hog_ptr, cell_hist_ptr, copy_data_length);
        count += copy_length;
      }
    }
  }
  if (count != hog_hist_length) {
    std::cerr << "Hog histogram not aligned." << count << " " << hog_hist_length << std::endl;
    return CVI_FAILURE;
  }
  delete[] cell_histogram;
  Tracer::TraceEnd();
  Tracer::TraceBegin("Normalizing HOG histogram");
  hog_ptr = (float *)pstDstHist->pu8VirAddr;
  u32 &&block_data_length = block_length * pstHogCtrl->u8BinSize;
  u32 nums_of_block_feature = hog_hist_length / block_data_length;
  u32 neon_turn = block_data_length / 4;
  u32 neon_turn_left = neon_turn * 4;
  for (u32 i = 0; i < nums_of_block_feature; i++) {
    float count_total = 0;
    auto &&skip_i = i * block_data_length;
    float *block_head = hog_ptr + skip_i;
    for (u32 j = 0; j < neon_turn; j++) {
      float32x4_t f = vld1q_f32(block_head);
      float32x4_t result = vmulq_f32(f, f);
      count_total += vaddvq_f32(result);
      block_head += 4;
    }
    for (u32 j = neon_turn_left; j < block_data_length; j++) {
      count_total += hog_ptr[skip_i + j] * hog_ptr[skip_i + j];
    }
    float count = 1.f / sqrt(count_total);
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
      if (pstDstMag == NULL) {
        std::cerr << "Under mode IVE_MAG_AND_ANG_OUT_CTRL_MAG Magnitude image cannot be NULL."
                  << std::endl;
        return CVI_FAILURE;
      }
      handle_ctx->t_h.t_magandang.exportOption(true, false);
      outputs.emplace_back(*cpp_dst);
    } break;
    case IVE_MAG_AND_ANG_OUT_CTRL_ANG: {
      if (pstDstAng == NULL) {
        std::cerr << "Under mode IVE_MAG_AND_ANG_OUT_CTRL_ANG angle image cannot be NULL."
                  << std::endl;
        return CVI_FAILURE;
      }
      handle_ctx->t_h.t_magandang.exportOption(false, true);
      outputs.emplace_back(*cpp_dst2);
    } break;
    case IVE_MAG_AND_ANG_OUT_CTRL_MAG_AND_ANG: {
      if (pstDstMag == NULL || pstDstAng == NULL) {
        std::cerr << "Under mode IVE_MAG_AND_ANG_OUT_CTRL_MAG_AND_ANG both outputs cannot be NULL."
                  << std::endl;
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
  handle_ctx->t_h.t_magandang.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  handle_ctx->t_h.t_magandang.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Map(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_MEM_INFO_S *pstMap,
                    IVE_DST_IMAGE_S *pstDst, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  auto &shape = handle_ctx->t_h.t_tblmgr.getTblTLShape(FMT_U8);
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
  handle_ctx->t_h.t_tbl.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  handle_ctx->t_h.t_tbl.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_NormGrad(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDstH,
                         IVE_DST_IMAGE_S *pstDstV, IVE_DST_IMAGE_S *pstDstHV,
                         IVE_NORM_GRAD_CTRL_S *pstNormGradCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  int npu_num = handle_ctx->t_h.t_sobel_gradonly.getNpuNum();
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs;
  if (pstNormGradCtrl->enOutCtrl == IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER) {
    IVE_IMAGE_S dstH_BF16, dstV_BF16;
    CVI_IVE_CreateImage(pIveHandle, &dstH_BF16, IVE_IMAGE_TYPE_BF16C1, pstSrc->u16Width,
                        pstSrc->u16Height);
    CVI_IVE_CreateImage(pIveHandle, &dstV_BF16, IVE_IMAGE_TYPE_BF16C1, pstSrc->u16Width,
                        pstSrc->u16Height);
    CviImg *cpp_dstv = reinterpret_cast<CviImg *>(dstV_BF16.tpu_block);
    CviImg *cpp_dsth = reinterpret_cast<CviImg *>(dstH_BF16.tpu_block);
    outputs.emplace_back(*cpp_dstv);
    outputs.emplace_back(*cpp_dsth);
    IveKernel kernel_w = createKernel(&handle_ctx->ctx, npu_num, 3, 3, IVE_KERNEL::SOBEL_X);
    IveKernel kernel_h = createKernel(&handle_ctx->ctx, npu_num, 3, 3, IVE_KERNEL::SOBEL_Y);
    handle_ctx->t_h.t_sobel_gradonly.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_sobel_gradonly.setKernel(kernel_w, kernel_h);
    handle_ctx->t_h.t_sobel_gradonly.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
    kernel_h.img.Free(&handle_ctx->ctx);
    IVE_ITC_CRTL_S iveItcCtrl;
    iveItcCtrl.enType = IVE_ITC_NORMALIZE;
    CVI_IVE_ImageTypeConvert(pIveHandle, &dstV_BF16, pstDstV, &iveItcCtrl, 0);
    CVI_IVE_ImageTypeConvert(pIveHandle, &dstH_BF16, pstDstH, &iveItcCtrl, 0);
    CVI_SYS_FreeI(pIveHandle, &dstV_BF16);
    CVI_SYS_FreeI(pIveHandle, &dstH_BF16);
  } else if (pstNormGradCtrl->enOutCtrl == IVE_NORM_GRAD_OUT_CTRL_HOR) {
    IVE_IMAGE_S dst_BF16;
    CVI_IVE_CreateImage(pIveHandle, &dst_BF16, IVE_IMAGE_TYPE_BF16C1, pstSrc->u16Width,
                        pstSrc->u16Height);
    CviImg *cpp_dsth = reinterpret_cast<CviImg *>(dst_BF16.tpu_block);
    outputs.emplace_back(*cpp_dsth);
    IveKernel kernel_h = createKernel(&handle_ctx->ctx, npu_num, 3, 3, IVE_KERNEL::SOBEL_Y);
    handle_ctx->t_h.t_filter_bf16.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_filter_bf16.setKernel(kernel_h);
    handle_ctx->t_h.t_filter_bf16.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
    kernel_h.img.Free(&handle_ctx->ctx);
    IVE_ITC_CRTL_S iveItcCtrl;
    iveItcCtrl.enType = IVE_ITC_NORMALIZE;
    CVI_IVE_ImageTypeConvert(pIveHandle, &dst_BF16, pstDstH, &iveItcCtrl, 0);
    CVI_SYS_FreeI(pIveHandle, &dst_BF16);
  } else if (pstNormGradCtrl->enOutCtrl == IVE_NORM_GRAD_OUT_CTRL_VER) {
    IVE_IMAGE_S dst_BF16;
    CVI_IVE_CreateImage(pIveHandle, &dst_BF16, IVE_IMAGE_TYPE_BF16C1, pstSrc->u16Width,
                        pstSrc->u16Height);
    CviImg *cpp_dstv = reinterpret_cast<CviImg *>(dst_BF16.tpu_block);
    outputs.emplace_back(*cpp_dstv);
    IveKernel kernel_w = createKernel(&handle_ctx->ctx, npu_num, 3, 3, IVE_KERNEL::SOBEL_X);
    handle_ctx->t_h.t_filter_bf16.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_filter_bf16.setKernel(kernel_w);
    handle_ctx->t_h.t_filter_bf16.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
    IVE_ITC_CRTL_S iveItcCtrl;
    iveItcCtrl.enType = IVE_ITC_NORMALIZE;
    CVI_IVE_ImageTypeConvert(pIveHandle, &dst_BF16, pstDstV, &iveItcCtrl, 0);
    CVI_SYS_FreeI(pIveHandle, &dst_BF16);
  } else if (pstNormGradCtrl->enOutCtrl == IVE_NORM_GRAD_OUT_CTRL_COMBINE) {
    IVE_IMAGE_S dst_BF16;
    CVI_IVE_CreateImage(pIveHandle, &dst_BF16, IVE_IMAGE_TYPE_BF16C1, pstSrc->u16Width,
                        pstSrc->u16Height);
    CviImg *cpp_dsthv = reinterpret_cast<CviImg *>(dst_BF16.tpu_block);
    outputs.emplace_back(*cpp_dsthv);
    IveKernel kernel_w = createKernel(&handle_ctx->ctx, npu_num, 3, 3, IVE_KERNEL::SOBEL_X);
    IveKernel kernel_h = createKernel(&handle_ctx->ctx, npu_num, 3, 3, IVE_KERNEL::SOBEL_Y);
    handle_ctx->t_h.t_sobel.setTblMgr(&handle_ctx->t_h.t_tblmgr);
    handle_ctx->t_h.t_sobel.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_sobel.setKernel(kernel_w, kernel_h);
    handle_ctx->t_h.t_sobel.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
    kernel_h.img.Free(&handle_ctx->ctx);
    IVE_ITC_CRTL_S iveItcCtrl;
    iveItcCtrl.enType = IVE_ITC_NORMALIZE;
    CVI_IVE_ImageTypeConvert(pIveHandle, &dst_BF16, pstDstHV, &iveItcCtrl, 0);
    CVI_SYS_FreeI(pIveHandle, &dst_BF16);
  } else {
    return CVI_FAILURE;
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Or(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                   IVE_DST_IMAGE_S *pstDst, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_or.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs = {*cpp_dst};

  handle_ctx->t_h.t_or.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Sigmoid(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                        bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_add.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};
  handle_ctx->t_h.t_sig.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  handle_ctx->t_h.t_sig.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  return BM_SUCCESS;
}

CVI_S32 CVI_IVE_SAD(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstSad, IVE_DST_IMAGE_S *pstThr, IVE_SAD_CTRL_S *pstSadCtrl,
                    bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
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
  handle_ctx->t_h.t_sad.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  handle_ctx->t_h.t_sad.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  if (!is_output_u8) {
    IVE_ITC_CRTL_S iveItcCtrl;
    iveItcCtrl.enType = IVE_ITC_SATURATE;
    ret = CVI_IVE_ImageTypeConvert(pIveHandle, &dst_BF16, pstSad, &iveItcCtrl, 0);
    CVI_SYS_FreeI(pIveHandle, &dst_BF16);
  }
  return ret;
}

CVI_S32 CVI_IVE_Sobel(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDstH,
                      IVE_DST_IMAGE_S *pstDstV, IVE_SOBEL_CTRL_S *pstSobelCtrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_sobel.setTblMgr(&handle_ctx->t_h.t_tblmgr);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dsth = reinterpret_cast<CviImg *>(pstDstH->tpu_block);
  CviImg *cpp_dstv = reinterpret_cast<CviImg *>(pstDstV->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs;
  u8 mask_sz = pstSobelCtrl->u8MaskSize;
  if (pstSobelCtrl->enOutCtrl == IVE_SOBEL_OUT_CTRL_BOTH) {
    int npu_num = handle_ctx->t_h.t_sobel_gradonly.getNpuNum();
    outputs.emplace_back(*cpp_dstv);
    outputs.emplace_back(*cpp_dsth);
    IveKernel kernel_w =
        createKernel(&handle_ctx->ctx, npu_num, mask_sz, mask_sz, IVE_KERNEL::SOBEL_X);
    IveKernel kernel_h =
        createKernel(&handle_ctx->ctx, npu_num, mask_sz, mask_sz, IVE_KERNEL::SOBEL_Y);
    handle_ctx->t_h.t_sobel_gradonly.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_sobel_gradonly.setKernel(kernel_w, kernel_h);
    handle_ctx->t_h.t_sobel_gradonly.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
    kernel_h.img.Free(&handle_ctx->ctx);
  } else if (pstSobelCtrl->enOutCtrl == IVE_SOBEL_OUT_CTRL_HOR) {
    outputs.emplace_back(*cpp_dsth);
    IveKernel kernel_h = createKernel(&handle_ctx->ctx, cpp_src->m_tg.shape.c, mask_sz, mask_sz,
                                      IVE_KERNEL::SOBEL_Y);
    handle_ctx->t_h.t_filter_bf16.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_filter_bf16.setKernel(kernel_h);
    handle_ctx->t_h.t_filter_bf16.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
    kernel_h.img.Free(&handle_ctx->ctx);
  } else if (pstSobelCtrl->enOutCtrl == IVE_SOBEL_OUT_CTRL_VER) {
    outputs.emplace_back(*cpp_dstv);
    IveKernel kernel_w = createKernel(&handle_ctx->ctx, cpp_src->m_tg.shape.c, mask_sz, mask_sz,
                                      IVE_KERNEL::SOBEL_X);
    handle_ctx->t_h.t_filter_bf16.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_filter_bf16.setKernel(kernel_w);
    handle_ctx->t_h.t_filter_bf16.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
  } else {
    return CVI_FAILURE;
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Sub(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, IVE_SUB_CTRL_S *ctrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  int ret = CVI_FAILURE;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  if (ctrl->enMode == IVE_SUB_MODE_NORMAL) {
    ret = CVI_SUCCESS;
    handle_ctx->t_h.t_sub.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
    CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
    CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
    std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
    std::vector<CviImg> outputs = {*cpp_dst};

    handle_ctx->t_h.t_sub.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  } else if (ctrl->enMode == IVE_SUB_MODE_ABS) {
    ret = CVI_SUCCESS;
    handle_ctx->t_h.t_sub_abs.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
    CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
    CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
    std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
    std::vector<CviImg> outputs = {*cpp_dst};

    handle_ctx->t_h.t_sub_abs.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  }
  return ret;
}

CVI_S32 CVI_IVE_Thresh(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_THRESH_CTRL_S *ctrl, bool bInstant) {
  ScopedTrace t(__PRETTY_FUNCTION__);
  int ret = CVI_FAILURE;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};

  if (ctrl->enMode == IVE_THRESH_MODE_BINARY) {
    ret = CVI_SUCCESS;
    if (ctrl->u8MinVal == 0 && ctrl->u8MaxVal == 255) {
      handle_ctx->t_h.t_thresh.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
      handle_ctx->t_h.t_thresh.setThreshold(ctrl->u8LowThr);
      handle_ctx->t_h.t_thresh.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
    } else {
      handle_ctx->t_h.t_thresh_hl.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
      handle_ctx->t_h.t_thresh_hl.setThreshold(ctrl->u8LowThr, ctrl->u8MinVal, ctrl->u8MaxVal);
      handle_ctx->t_h.t_thresh_hl.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
    }
  }
  return ret;
}

CVI_S32 CVI_IVE_Thresh_S16(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                           IVE_THRESH_S16_CTRL_S *pstThrS16Ctrl, bool bInstant) {
  if (pstSrc->enType != IVE_IMAGE_TYPE_S16C1) {
    std::cerr << "Input only accepts S16C1 image format." << std::endl;
    return CVI_FAILURE;
  }
  CVI_IVE_BufRequest(pIveHandle, pstSrc);
  CVI_IVE_BufRequest(pIveHandle, pstDst);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  u64 data_size = cpp_src->m_tg.stride.n / getFmtSize(cpp_src->m_tg.fmt);
  if (pstThrS16Ctrl->enMode == IVE_THRESH_S16_MODE_S16_TO_S8_MIN_MID_MAX ||
      pstThrS16Ctrl->enMode == IVE_THRESH_S16_MODE_S16_TO_S8_MIN_ORI_MAX) {
    if (pstDst->enType != IVE_IMAGE_TYPE_S8C1) {
      std::cerr << "Output only accepts S8C1 image format." << std::endl;
      return CVI_FAILURE;
    }
    bool is_mmm =
        (pstThrS16Ctrl->enMode == IVE_THRESH_S16_MODE_S16_TO_S8_MIN_MID_MAX) ? true : false;
    neonS162S8ThresholdLH((s16 *)pstSrc->pu8VirAddr[0], (s8 *)pstDst->pu8VirAddr[0], data_size,
                          pstThrS16Ctrl->s16LowThr, pstThrS16Ctrl->s16HighThr,
                          pstThrS16Ctrl->un8MinVal.s8Val, pstThrS16Ctrl->un8MidVal.s8Val,
                          pstThrS16Ctrl->un8MaxVal.s8Val, is_mmm);
  } else {
    if (pstDst->enType != IVE_IMAGE_TYPE_U8C1) {
      std::cerr << "Output only accepts U8C1 image format." << std::endl;
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
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_xor.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs = {*cpp_dst};

  handle_ctx->t_h.t_xor.run(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  return CVI_SUCCESS;
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
inline void GetGrayIntegralImage(u8 *Src, u32 *Integral, int Width, int Height, int Stride) {
  memset(Integral, 0, (Width + 1) * sizeof(uint));
  for (int Y = 0; Y < Height; Y++) {
    u8 *LinePS = Src + Y * Stride;
    u32 *LinePL = Integral + Y * (Width + 1) + 1;
    u32 *LinePD = Integral + (Y + 1) * (Width + 1) + 1;
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

inline int cal_hist(int cols, int rows, u8 *image, u32 *hist, int Stride, int num_bins) {
  int col, row;
  if (cols < 1 || rows < 1 || num_bins < 1) {
    return (1);
  }
  memset(hist, 0, sizeof(u32) * num_bins);

  for (int Y = 0; Y < rows; Y++) {
    u8 *LinePS = image + Y * cols;
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
    eqhist[i] = round((((float)curr) * 255) / total);
  }
  return (0);
}

inline int histogramEqualisation(int cols, int rows, int stride, u8 *image, u8 *pDst) {
  u32 hist[256] = {0};
  u8 new_gray_level[256] = {0};
  int col, row, total, st;

  st = cal_hist(cols, rows, image, hist, stride, 256);
  if (st > 0) {
    return (st);
  }
  total = cols * rows;
  st = equalize_hist(hist, new_gray_level, total, 256);
  if (st > 0) {
    return (st);
  }
  for (row = 0; row < rows; row++) {
    for (col = 0; col < cols; col++) {
      pDst[col] = (unsigned char)new_gray_level[pDst[col]];
    }
    pDst += cols;
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

  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  u64 data_size = cpp_src->m_tg.stride.n / getFmtSize(cpp_src->m_tg.fmt);
  u32 *ptr = (u32 *)pstDst->pu8VirAddr;
  GetGrayIntegralImage((u8 *)pstSrc->pu8VirAddr[0], ptr, (int)pstSrc->u16Width,
                       (int)pstSrc->u16Height, (int)data_size);

  CVI_IVE_BufFlush(pIveHandle, pstSrc);
  return CVI_SUCCESS;
}

CVI_S32 CVI_IVE_Hist(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_MEM_INFO_S *pstDst,
                     bool bInstant) {
  if (pstSrc->enType != IVE_IMAGE_TYPE_U8C1) {
    std::cerr << "Output only accepts U8C1 image format." << std::endl;
    return CVI_FAILURE;
  }

  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  u64 data_size = cpp_src->m_tg.stride.n / getFmtSize(cpp_src->m_tg.fmt);
  cal_hist((int)pstSrc->u16Width, (int)pstSrc->u16Height, (u8 *)pstSrc->pu8VirAddr[0],
           (u32 *)pstDst->pu8VirAddr, (int)data_size, 256);
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

  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  u64 data_size = cpp_src->m_tg.stride.n / getFmtSize(cpp_src->m_tg.fmt);

  histogramEqualisation((int)pstSrc->u16Width, (int)pstSrc->u16Height, (int)data_size,
                        (u8 *)pstSrc->pu8VirAddr[0], (u8 *)pstDst->pu8VirAddr[0]);

  CVI_IVE_BufFlush(pIveHandle, pstSrc);

  return CVI_SUCCESS;
}

double cal_norm_cc(unsigned char *psrc1, unsigned char *psrc2, int srcw, int srch) {
  int i, j, wxh;
  uint t1, t2, t3;

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

  return ((double)t1 / t2 * ((double)(1.0) / t3));
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
  // u32 *ptr = (u32 *)pstDst->pu8VirAddr[0];
  double rt = cal_norm_cc((u8 *)pstSrc1->pu8VirAddr[0], (u8 *)pstSrc2->pu8VirAddr[0],
                          (int)pstSrc1->u16Width, (int)pstSrc1->u16Height);

  CVI_IVE_BufFlush(pIveHandle, pstSrc1);
  CVI_IVE_BufFlush(pIveHandle, pstSrc2);

  return CVI_SUCCESS;
}

inline void uint16_8bit(uint16_t *in, uint8_t *out, int dim) {
  int i;

  for (i = 0; i < dim; i++) {
    uint16_t n = in[i];  // because shifting the sign bit invokes UB
    uint8_t hi = ((n >> 8) & 0xff);
    uint8_t lo = ((n >> 0) & 0xff);
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

