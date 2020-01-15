#include "ive.h"
#include <glog/logging.h>
#include "kernel_generator.hpp"
#include "tpu_data.hpp"

#include "tpu/tpu_add.hpp"
#include "tpu/tpu_and.hpp"
#include "tpu/tpu_filter.hpp"
#include "tpu/tpu_morph.hpp"
#include "tpu/tpu_or.hpp"
#include "tpu/tpu_sobel.hpp"
#include "tpu/tpu_sub.hpp"
#include "tpu/tpu_threshold.hpp"
#include "tpu/tpu_xor.hpp"

struct TPU_HANDLE {
  IveTPUAdd t_add;
  IveTPUAnd t_and;
  IveTPUErode t_erode;
  IveTPUFilter t_filter;
  IveTPUFilterBF16 t_filter_bf16;
  IveTPUOr t_or;
  IveTPUSobelGradOnly t_sobel_gradonly;
  IveTPUSobel t_sobel;
  IveTPUSub t_sub;
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
  return (void *)handle_ctx;
}

CVI_S32 CVI_IVE_DestroyHandle(IVE_HANDLE pIveHandle) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  destroyHandle(&handle_ctx->ctx);
  return 0;
}

CVI_S32 CVI_IVE_CreateImage(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg, IVE_IMAGE_TYPE_E enType,
                            u16 u16Width, u16 u16Height) {
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
    case IVE_IMAGE_TYPE_FP32C1:
      fmt_size = 4;
      fmt = FMT_F32;
      break;
    default:
      std::cerr << "Not supported enType " << enType << std::endl;
      return 1;
      break;
  }
  int total_size = c * u16Width * u16Height * fmt_size;
  pstImg->tpu_block =
      reinterpret_cast<CVI_IMG *>(new CviImg(&handle_ctx->ctx, c, u16Height, u16Width, fmt));
  auto *cpp_img = reinterpret_cast<CviImg *>(pstImg->tpu_block);

  pstImg->enType = enType;
  pstImg->u16Width = cpp_img->m_tg.shape.w;
  pstImg->u16Height = cpp_img->m_tg.shape.h;
  pstImg->u16Reserved = fmt_size;

  int img_sz = pstImg->u16Width * pstImg->u16Height * fmt_size;
  for (size_t i = 0; i < cpp_img->m_tg.shape.c; i++) {
    pstImg->pu8VirAddr[i] = cpp_img->GetVAddr() + i * img_sz;
    pstImg->u64PhyAddr[i] = cpp_img->GetPAddr() + i * img_sz;
    pstImg->u16Stride[i] = pstImg->u16Width * fmt_size;
  }

  for (size_t i = cpp_img->m_tg.shape.c; i < 3; i++) {
    pstImg->pu8VirAddr[i] = NULL;
    pstImg->u64PhyAddr[i] = -1;
    pstImg->u16Stride[i] = 0;
  }
  return 0;
}

CVI_S32 CVI_SYS_Free(IVE_HANDLE pIveHandle, IVE_IMAGE_S *pstImg) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  auto *cpp_img = reinterpret_cast<CviImg *>(pstImg->tpu_block);
  cpp_img->Free(&handle_ctx->ctx);
  delete cpp_img;
  return 0;
}

CVI_S32 CVI_IVE_DMA(IVE_HANDLE pIveHandle, IVE_DATA_S *pstSrc, IVE_DST_DATA_S *pstDst,
                    IVE_DMA_CTRL_S *pstDmaCtrl, bool bInstant) {
  int ret = 1;
  if (pstDmaCtrl->enMode == IVE_DMA_MODE_DIRECT_COPY) {
    ret = 0;
    int size = pstSrc->u16Stride * pstSrc->u16Height * pstSrc->u16Reserved;
    memcpy(pstDst->pu8VirAddr, pstSrc->pu8VirAddr, size);
  }
  return ret;
}

static inline float convert_bf16_fp32(u16 bf16) {
  float fp32;
  u16 *fparr = (u16 *)&fp32;
  fparr[1] = bf16;
  fparr[0] = 0;
  return fp32;
}

CVI_S32 CVI_IVE_ImageTypeConvert(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc,
                                 IVE_DST_IMAGE_S *pstDst, IVE_ITC_CRTL_S *pstItcCtrl,
                                 bool bInstant) {
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  if (cpp_src->m_tg.fmt == FMT_BF16 && cpp_dst->m_tg.fmt == FMT_F32) {
    u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
    float *dst_ptr = (float *)cpp_dst->GetVAddr();

    for (u64 i = 0; i < cpp_src->GetImgSize(); i++) {
      dst_ptr[i] = convert_bf16_fp32(src_ptr[i]);
    }
  } else if (cpp_src->m_tg.fmt == FMT_BF16 &&
             (cpp_dst->m_tg.fmt == FMT_U8 || cpp_dst->m_tg.fmt == FMT_I8)) {
    u16 *src_ptr = (u16 *)cpp_src->GetVAddr();
    u8 *dst_ptr = (u8 *)cpp_dst->GetVAddr();
    if (pstItcCtrl->enType == IVE_ITC_NORMALIZE) {
      float *tmp_arr = new float[cpp_src->GetImgSize()];
      float min = 10000000.f, max = -100000000.f;
      u64 img_size = cpp_src->m_tg.shape.c * cpp_src->m_tg.shape.h * cpp_src->m_tg.shape.w;
      for (u64 i = 0; i < img_size; i++) {
        tmp_arr[i] = convert_bf16_fp32(src_ptr[i]);
        if (tmp_arr[i] < min) {
          min = tmp_arr[i];
        }
        if (tmp_arr[i] > max) {
          max = tmp_arr[i];
        }
      }
      float multiplier = 255.f / (max - min);
      short s8_offset = cpp_dst->m_tg.fmt == FMT_U8 ? 0 : 128;
      for (u64 i = 0; i < img_size; i++) {
        dst_ptr[i] = (u8)(multiplier * (tmp_arr[i] - min)) - s8_offset;
      }
      delete[] tmp_arr;
    } else if (pstItcCtrl->enType == IVE_ITC_SATURATE) {
      for (u64 i = 0; i < cpp_src->GetImgSize(); i++) {
        dst_ptr[i] = convert_bf16_fp32(src_ptr[i]);
      }
    } else {
      return 1;
    }
  } else {
    return 1;
  }
  return 0;
}

CVI_S32 CVI_IVE_Add(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, IVE_ADD_CTRL_S *ctrl, bool bInstant) {
  int ret = 1;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);

  float *x = (float *)&ctrl->u0q16X;
  float *y = (float *)&ctrl->u0q16Y;
  if ((*x == 1 && *y == 1) || (*x == 0 && *y == 0)) {
    ret = 0;
    handle_ctx->t_h.t_add.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
    CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
    CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
    std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
    std::vector<CviImg> outputs = {*cpp_dst};

    handle_ctx->t_h.t_add.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs,
                                              &outputs);
  }
  return ret;
}

CVI_S32 CVI_IVE_And(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, bool bInstant) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_and.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs = {*cpp_dst};

  handle_ctx->t_h.t_and.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  return 0;
}

CVI_S32 CVI_IVE_Dilate(IVE_HANDLE *pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_DILATE_CTRL_S *pstDilateCtrl, bool bInstant) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_filter.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};

  CviImg cimg(&handle_ctx->ctx, cpp_src->m_tg.shape.c, 3, 3, FMT_U8);
  IveKernel kernel;
  kernel.img = cimg;
  kernel.img.GetVAddr();
  for (size_t i = 0; i < cpp_src->m_tg.shape.c; i++) {
    memcpy(kernel.img.GetVAddr() + i * 9, pstDilateCtrl->au8Mask, 9);
  }
  kernel.multiplier.f = 1.f;
  QuantizeMultiplierSmallerThanOne(kernel.multiplier.f, &kernel.multiplier.base,
                                   &kernel.multiplier.shift);
  handle_ctx->t_h.t_filter.setKernel(kernel);
  handle_ctx->t_h.t_filter.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs,
                                               &outputs);
  kernel.img.Free(&handle_ctx->ctx);
  return 0;
}

CVI_S32 CVI_IVE_Erode(IVE_HANDLE *pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                      IVE_ERODE_CTRL_S *pstErodeCtrl, bool bInstant) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_erode.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};

  CviImg cimg(&handle_ctx->ctx, cpp_src->m_tg.shape.c, 3, 3, FMT_U8);
  IveKernel kernel;
  kernel.img = cimg;
  kernel.img.GetVAddr();
  for (size_t i = 0; i < cpp_src->m_tg.shape.c; i++) {
    memcpy(kernel.img.GetVAddr() + i * 9, pstErodeCtrl->au8Mask, 9);
  }
  kernel.multiplier.f = 1.f;
  QuantizeMultiplierSmallerThanOne(kernel.multiplier.f, &kernel.multiplier.base,
                                   &kernel.multiplier.shift);
  handle_ctx->t_h.t_erode.setKernel(kernel);
  handle_ctx->t_h.t_erode.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs,
                                              &outputs);
  kernel.img.Free(&handle_ctx->ctx);
  return 0;
}

CVI_S32 CVI_IVE_Filter(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                       IVE_FILTER_CTRL_S *pstFltCtrl, bool bInstant) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_filter.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs = {*cpp_dst};

  CviImg cimg(&handle_ctx->ctx, cpp_src->m_tg.shape.c, 3, 3, FMT_I8);
  IveKernel kernel;
  kernel.img = cimg;
  kernel.img.GetVAddr();
  for (size_t i = 0; i < cpp_src->m_tg.shape.c; i++) {
    memcpy(kernel.img.GetVAddr() + i * 9, pstFltCtrl->as8Mask, 9);
  }
  kernel.multiplier.f = 1.f / pstFltCtrl->u8Norm;
  QuantizeMultiplierSmallerThanOne(kernel.multiplier.f, &kernel.multiplier.base,
                                   &kernel.multiplier.shift);
  handle_ctx->t_h.t_filter.setKernel(kernel);
  handle_ctx->t_h.t_filter.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs,
                                               &outputs);
  kernel.img.Free(&handle_ctx->ctx);
  return 0;
}

CVI_S32 CVI_IVE_Or(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                   IVE_DST_IMAGE_S *pstDst, bool bInstant) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_or.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs = {*cpp_dst};

  handle_ctx->t_h.t_or.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  return 0;
}

CVI_S32 CVI_IVE_Sobel(IVE_HANDLE *pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDstH,
                      IVE_DST_IMAGE_S *pstDstV, IVE_SOBEL_CTRL_S *pstSobelCtrl, bool bInstant) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  CviImg *cpp_src = reinterpret_cast<CviImg *>(pstSrc->tpu_block);
  CviImg *cpp_dsth = reinterpret_cast<CviImg *>(pstDstH->tpu_block);
  CviImg *cpp_dstv = reinterpret_cast<CviImg *>(pstDstV->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src};
  std::vector<CviImg> outputs;
  if (pstSobelCtrl->enOutCtrl == IVE_SOBEL_OUT_CTRL_BOTH) {
    outputs.emplace_back(*cpp_dstv);
    outputs.emplace_back(*cpp_dsth);
    IveKernel kernel_w =
        createKernel(&handle_ctx->ctx, cpp_src->m_tg.shape.c, 3, 3, IVE_KERNEL::SOBEL_X);
    IveKernel kernel_h =
        createKernel(&handle_ctx->ctx, cpp_src->m_tg.shape.c, 3, 3, IVE_KERNEL::SOBEL_Y);
    handle_ctx->t_h.t_sobel_gradonly.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_sobel_gradonly.setKernel(kernel_w, kernel_h);
    handle_ctx->t_h.t_sobel_gradonly.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx,
                                                         inputs, &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
    kernel_h.img.Free(&handle_ctx->ctx);
  } else if (pstSobelCtrl->enOutCtrl == IVE_SOBEL_OUT_CTRL_HOR) {
    outputs.emplace_back(*cpp_dsth);
    IveKernel kernel_h =
        createKernel(&handle_ctx->ctx, cpp_src->m_tg.shape.c, 3, 3, IVE_KERNEL::SOBEL_Y);
    handle_ctx->t_h.t_filter_bf16.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_filter_bf16.setKernel(kernel_h);
    handle_ctx->t_h.t_filter_bf16.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs,
                                                      &outputs);
    kernel_h.img.Free(&handle_ctx->ctx);
  } else if (pstSobelCtrl->enOutCtrl == IVE_SOBEL_OUT_CTRL_VER) {
    outputs.emplace_back(*cpp_dstv);
    IveKernel kernel_w =
        createKernel(&handle_ctx->ctx, cpp_src->m_tg.shape.c, 3, 3, IVE_KERNEL::SOBEL_X);
    handle_ctx->t_h.t_filter_bf16.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    handle_ctx->t_h.t_filter_bf16.setKernel(kernel_w);
    handle_ctx->t_h.t_filter_bf16.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs,
                                                      &outputs);
    kernel_w.img.Free(&handle_ctx->ctx);
  } else {
    return 1;
  }
  return 0;
}

CVI_S32 CVI_IVE_Sub(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, IVE_SUB_CTRL_S *ctrl, bool bInstant) {
  int ret = 1;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  if (ctrl->enMode == IVE_SUB_MODE_BUTT) {
    ret = 0;
    handle_ctx->t_h.t_sub.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
    CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
    CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
    CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
    std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
    std::vector<CviImg> outputs = {*cpp_dst};

    handle_ctx->t_h.t_sub.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs,
                                              &outputs);
  }
  return ret;
}

CVI_S32 CVI_IVE_Thresh(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_DST_IMAGE_S *pstDst,
                       IVE_THRESH_CTRL_S *ctrl, bool bInstant) {
  int ret = 1;
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src1};
  std::vector<CviImg> outputs = {*cpp_dst};

  if (ctrl->enMode == IVE_THRESH_MODE_BINARY) {
    ret = 0;
    if (ctrl->u8MinVal == 0 && ctrl->u8MaxVal == 255) {
      handle_ctx->t_h.t_thresh.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
      handle_ctx->t_h.t_thresh.setThreshold(ctrl->u8LowThr);
      handle_ctx->t_h.t_thresh.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs,
                                                   &outputs);
    } else {
      handle_ctx->t_h.t_thresh_hl.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
      handle_ctx->t_h.t_thresh_hl.setThreshold(ctrl->u8LowThr, ctrl->u8MinVal, ctrl->u8MaxVal);
      handle_ctx->t_h.t_thresh_hl.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs,
                                                      &outputs);
    }
  }
  return ret;
}

CVI_S32 CVI_IVE_Xor(IVE_HANDLE pIveHandle, IVE_SRC_IMAGE_S *pstSrc1, IVE_SRC_IMAGE_S *pstSrc2,
                    IVE_DST_IMAGE_S *pstDst, bool bInstant) {
  IVE_HANDLE_CTX *handle_ctx = reinterpret_cast<IVE_HANDLE_CTX *>(pIveHandle);
  handle_ctx->t_h.t_xor.init(&handle_ctx->ctx, handle_ctx->bk_ctx);
  CviImg *cpp_src1 = reinterpret_cast<CviImg *>(pstSrc1->tpu_block);
  CviImg *cpp_src2 = reinterpret_cast<CviImg *>(pstSrc2->tpu_block);
  CviImg *cpp_dst = reinterpret_cast<CviImg *>(pstDst->tpu_block);
  std::vector<CviImg> inputs = {*cpp_src1, *cpp_src2};
  std::vector<CviImg> outputs = {*cpp_dst};

  handle_ctx->t_h.t_xor.runSingleSizeKernel(&handle_ctx->ctx, handle_ctx->bk_ctx, inputs, &outputs);
  return 0;
}
