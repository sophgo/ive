#include "bmkernel/bm_kernel.h"
#include "ive.h"

#include <math.h>
#include <stdio.h>
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"

int cpu_ref(const int channels, IVE_SRC_IMAGE_S *src, IVE_DST_IMAGE_S *dstH, IVE_DST_IMAGE_S *dstV,
            IVE_DST_IMAGE_S *dstMag, IVE_DST_IMAGE_S *dstAng);

int main(int argc, char **argv) {
  CVI_SYS_LOGGING(argv[0]);
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  // Fetch image information
  IplImage *img = cvLoadImage("cat.png", 0);
  IVE_SRC_IMAGE_S src;
  CVI_IVE_CreateImage(handle, &src, IVE_IMAGE_TYPE_U8C1, img->width, img->height);
  memcpy(src.pu8VirAddr[0], img->imageData, img->nChannels * img->width * img->height);

  IVE_DST_IMAGE_S dstH, dstV;
  CVI_IVE_CreateImage(handle, &dstV, IVE_IMAGE_TYPE_BF16C1, img->width, img->height);
  CVI_IVE_CreateImage(handle, &dstH, IVE_IMAGE_TYPE_BF16C1, img->width, img->height);

  IVE_DST_IMAGE_S dstH_u8, dstV_u8;
  CVI_IVE_CreateImage(handle, &dstH_u8, IVE_IMAGE_TYPE_U8C1, img->width, img->height);
  CVI_IVE_CreateImage(handle, &dstV_u8, IVE_IMAGE_TYPE_U8C1, img->width, img->height);

  IVE_DST_IMAGE_S dstMag, dstAng;
  CVI_IVE_CreateImage(handle, &dstMag, IVE_IMAGE_TYPE_BF16C1, img->width, img->height);
  CVI_IVE_CreateImage(handle, &dstAng, IVE_IMAGE_TYPE_BF16C1, img->width, img->height);

  IVE_DST_IMAGE_S dstMag_u8, dstAng_u8;
  CVI_IVE_CreateImage(handle, &dstMag_u8, IVE_IMAGE_TYPE_U8C1, img->width, img->height);
  CVI_IVE_CreateImage(handle, &dstAng_u8, IVE_IMAGE_TYPE_U8C1, img->width, img->height);

  printf("Run TPU Sobel Grad.\n");
  IVE_SOBEL_CTRL_S iveSblCtrl;
  iveSblCtrl.enOutCtrl = IVE_SOBEL_OUT_CTRL_BOTH;
  CVI_IVE_Sobel(handle, &src, &dstH, &dstV, &iveSblCtrl, 0);
  printf("Run TPU Mag and Ang.\n");
  IVE_MAG_AND_ANG_CTRL_S pstMaaCtrl;
  pstMaaCtrl.enOutCtrl = IVE_MAG_AND_ANG_OUT_CTRL_MAG_AND_ANG;
  pstMaaCtrl.no_negative = true;
  CVI_IVE_MagAndAng(handle, &dstH, &dstV, &dstMag, &dstAng, &pstMaaCtrl, 0);

  printf("Normalize result to 0-255.\n");
  IVE_ITC_CRTL_S iveItcCtrl;
  iveItcCtrl.enType = IVE_ITC_NORMALIZE;
  CVI_IVE_ImageTypeConvert(handle, &dstV, &dstV_u8, &iveItcCtrl, 0);
  CVI_IVE_ImageTypeConvert(handle, &dstH, &dstH_u8, &iveItcCtrl, 0);
  CVI_IVE_ImageTypeConvert(handle, &dstMag, &dstMag_u8, &iveItcCtrl, 0);
  CVI_IVE_ImageTypeConvert(handle, &dstAng, &dstAng_u8, &iveItcCtrl, 0);

  int ret = cpu_ref(img->nChannels, &src, &dstH, &dstV, &dstMag, &dstAng);

  // write result to disk
  printf("Save to image.\n");
  memcpy(img->imageData, dstV_u8.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_sobelV_c.png", img, 0);
  memcpy(img->imageData, dstH_u8.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_sobelH_c.png", img, 0);
  memcpy(img->imageData, dstMag_u8.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_mag_c.png", img, 0);
  memcpy(img->imageData, dstAng_u8.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_ang_c.png", img, 0);
  cvReleaseImage(&img);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dstH);
  CVI_SYS_FreeI(handle, &dstH_u8);
  CVI_SYS_FreeI(handle, &dstV);
  CVI_SYS_FreeI(handle, &dstV_u8);
  CVI_SYS_FreeI(handle, &dstMag);
  CVI_SYS_FreeI(handle, &dstMag_u8);
  CVI_SYS_FreeI(handle, &dstAng);
  CVI_SYS_FreeI(handle, &dstAng_u8);
  CVI_IVE_DestroyHandle(handle);

  return ret;
}

int cpu_ref(const int channels, IVE_SRC_IMAGE_S *src, IVE_DST_IMAGE_S *dstH, IVE_DST_IMAGE_S *dstV,
            IVE_DST_IMAGE_S *dstMag, IVE_DST_IMAGE_S *dstAng) {
  int ret = CVI_SUCCESS;
  u16 *dstH_ptr = (u16 *)dstH->pu8VirAddr[0];
  u16 *dstV_ptr = (u16 *)dstV->pu8VirAddr[0];
  u16 *dstMag_ptr = (u16 *)dstMag->pu8VirAddr[0];
  u16 *dstAng_ptr = (u16 *)dstAng->pu8VirAddr[0];
  float mul_val = 180.f / M_PI;
  float sqrt_epsilon = 0.01;
  float ang_epsilon = 0.02;  // atan2 0.02, mul => 1.3
  printf("Check Mag:\n");
  for (size_t i = 0; i < channels * src->u16Width * src->u16Height; i++) {
    float dstH_f = convert_bf16_fp32(dstH_ptr[i]);
    float dstV_f = convert_bf16_fp32(dstV_ptr[i]);
    float dstMag_f = convert_bf16_fp32(dstMag_ptr[i]);
    float sqrt_res = sqrtf(dstV_f * dstV_f + dstH_f * dstH_f);
    float error = fabs(sqrt_res - dstMag_f) / sqrt_res;
    if (error > sqrt_epsilon) {
      printf("[%lu] sqrt( %f, %f) = TPU %f, CPU %f. eplison = %f\n", i, dstV_f, dstH_f, dstMag_f,
             sqrt_res, error);
      ret = CVI_FAILURE;
    }
  }
  printf("Check Ang:\n");
  for (size_t i = 0; i < channels * src->u16Width * src->u16Height; i++) {
    float dstH_f = convert_bf16_fp32(dstH_ptr[i]);
    float dstV_f = convert_bf16_fp32(dstV_ptr[i]);
    float dstAng_f = convert_bf16_fp32(dstAng_ptr[i]);
    float atan2_res = (float)atan2(dstV_f, dstH_f) * mul_val;
    if (atan2_res < 0) {
      atan2_res += 360.f;
    }
    float error = fabs(atan2_res - dstAng_f) / atan2_res;
    if (error > ang_epsilon) {
      printf("[%lu] atan2( %f, %f) = TPU %f, CPU %f (%f). eplison = %f\n", i, dstV_f, dstH_f,
             dstAng_f, atan2_res, atan2_res - 360.f, error);
      ret = CVI_FAILURE;
    }
  }
  return ret;
}