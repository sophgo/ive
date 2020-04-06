#include "bmkernel/bm_kernel.h"
#include "bmtap2/1880v2_fp_convert.h"
#include "ive.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __ARM_ARCH
#include "arm_neon.h"
#endif

int cpu_ref(const int channels, IVE_SRC_IMAGE_S *src, IVE_DST_IMAGE_S *dstH, IVE_DST_IMAGE_S *dstV,
            IVE_DST_IMAGE_S *dstMag, IVE_DST_IMAGE_S *dstAng);

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Incorrect loop value. Usage: %s <file_name> <loop in value (1-1000)>\n", argv[0]);
    return CVI_FAILURE;
  }
  CVI_SYS_LOGGING(argv[0]);
  const char *file_name = argv[1];
  size_t total_run = atoi(argv[2]);
  printf("Loop value: %lu\n", total_run);
  if (total_run > 1000 || total_run == 0) {
    printf("Incorrect loop value. Usage: %s <file_name> <loop in value (1-1000)>\n", argv[0]);
    return CVI_FAILURE;
  }
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  // Fetch image information
  IVE_IMAGE_S src = CVI_IVE_ReadImage(handle, file_name, IVE_IMAGE_TYPE_U8C1);
  int nChannels = 1;
  int width = src.u16Width;
  int height = src.u16Height;

  IVE_DST_IMAGE_S dstH, dstV;
  CVI_IVE_CreateImage(handle, &dstV, IVE_IMAGE_TYPE_BF16C1, width, height);
  CVI_IVE_CreateImage(handle, &dstH, IVE_IMAGE_TYPE_BF16C1, width, height);

  IVE_DST_IMAGE_S dstH_u8, dstV_u8;
  CVI_IVE_CreateImage(handle, &dstH_u8, IVE_IMAGE_TYPE_U8C1, width, height);
  CVI_IVE_CreateImage(handle, &dstV_u8, IVE_IMAGE_TYPE_U8C1, width, height);

  IVE_DST_IMAGE_S dstMag, dstAng;
  CVI_IVE_CreateImage(handle, &dstMag, IVE_IMAGE_TYPE_BF16C1, width, height);
  CVI_IVE_CreateImage(handle, &dstAng, IVE_IMAGE_TYPE_BF16C1, width, height);

  IVE_DST_IMAGE_S dstMag_u8, dstAng_u8;
  CVI_IVE_CreateImage(handle, &dstMag_u8, IVE_IMAGE_TYPE_U8C1, width, height);
  CVI_IVE_CreateImage(handle, &dstAng_u8, IVE_IMAGE_TYPE_U8C1, width, height);

  printf("Run TPU Sobel Grad.\n");
  IVE_SOBEL_CTRL_S iveSblCtrl;
  iveSblCtrl.enOutCtrl = IVE_SOBEL_OUT_CTRL_BOTH;
  iveSblCtrl.u8MaskSize = 3;
  IVE_MAG_AND_ANG_CTRL_S pstMaaCtrl;
  pstMaaCtrl.enOutCtrl = IVE_MAG_AND_ANG_OUT_CTRL_MAG_AND_ANG;
  unsigned long total_s = 0;
  unsigned long total_mag = 0;
  struct timeval t0, t1, t2;
  for (size_t i = 0; i < total_run; i++) {
    gettimeofday(&t0, NULL);
    CVI_IVE_Sobel(handle, &src, &dstH, &dstV, &iveSblCtrl, 0);
    gettimeofday(&t1, NULL);
    CVI_IVE_MagAndAng(handle, &dstH, &dstV, &dstMag, &dstAng, &pstMaaCtrl, 0);
    gettimeofday(&t2, NULL);
    unsigned long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    unsigned long elapsed2 = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    total_s += elapsed;
    total_mag += elapsed2;
  }
  total_s /= total_run;
  total_mag /= total_run;

  printf("Normalize result to 0-255.\n");
  IVE_ITC_CRTL_S iveItcCtrl;
  iveItcCtrl.enType = IVE_ITC_NORMALIZE;
  CVI_IVE_ImageTypeConvert(handle, &dstV, &dstV_u8, &iveItcCtrl, 0);
  CVI_IVE_ImageTypeConvert(handle, &dstH, &dstH_u8, &iveItcCtrl, 0);
  CVI_IVE_ImageTypeConvert(handle, &dstMag, &dstMag_u8, &iveItcCtrl, 0);
  CVI_IVE_ImageTypeConvert(handle, &dstAng, &dstAng_u8, &iveItcCtrl, 0);

  CVI_IVE_BufRequest(handle, &src);
  CVI_IVE_BufRequest(handle, &dstH);
  CVI_IVE_BufRequest(handle, &dstV);
  CVI_IVE_BufRequest(handle, &dstMag);
  CVI_IVE_BufRequest(handle, &dstAng);
  int ret = cpu_ref(nChannels, &src, &dstH, &dstV, &dstMag, &dstAng);

  if (total_run == 1) {
    printf("TPU Sobel avg time %lu\n", total_s);
    printf("TPU MagAndAng avg time %lu\n", total_mag);
    // write result to disk
    printf("Save to image.\n");
    CVI_IVE_WriteImage(handle, "test_sobelV_c.png", &dstV_u8);
    CVI_IVE_WriteImage(handle, "test_sobelH_c.png", &dstH_u8);
    CVI_IVE_WriteImage(handle, "test_mag_c.png", &dstMag_u8);
    CVI_IVE_WriteImage(handle, "test_ang_c.png", &dstAng_u8);
  }
#ifdef __ARM_ARCH
  else {
    printf("OOO %-10s %10lu %10s %10s\n", "SOBEL GRAD", total_s, "NA", "NA");
    printf("OOO %-10s %10lu %10s %10s\n", "MagnAng", total_mag, "NA", "NA");
  }
#endif

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
  float sqrt_epsilon = 2;
  float ang_abs_limit = 1;
  printf("Check Mag:\n");
  for (size_t i = 0; i < channels * src->u16Width * src->u16Height; i++) {
    float dstH_f = convert_bf16_fp32(dstH_ptr[i]);
    float dstV_f = convert_bf16_fp32(dstV_ptr[i]);
    float dstMag_f = convert_bf16_fp32(dstMag_ptr[i]);
    float sqrt_res = sqrtf(dstV_f * dstV_f + dstH_f * dstH_f);
    float error = fabs(sqrt_res - dstMag_f);
    if (error > sqrt_epsilon) {
      printf("[%lu] sqrt( %f^2 + %f^2) = TPU %f, CPU %f. eplison = %f\n", i, dstV_f, dstH_f,
             dstMag_f, sqrt_res, error);
      ret = CVI_FAILURE;
    }
  }
  printf("Check Ang:\n");
  for (size_t i = 0; i < channels * src->u16Width * src->u16Height; i++) {
    float dstH_f = convert_bf16_fp32(dstH_ptr[i]);
    float dstV_f = convert_bf16_fp32(dstV_ptr[i]);
    float dstAng_f = convert_bf16_fp32(dstAng_ptr[i]);
    float atan2_res = (float)atan2(dstV_f, dstH_f) * mul_val;
    float error = fabs(atan2_res - dstAng_f);
    if (error > ang_abs_limit) {
      printf("[%lu] atan2( %f, %f) = TPU %f, CPU %f. eplison = %f\n", i, dstV_f, dstH_f, dstAng_f,
             atan2_res, error);
      ret = CVI_FAILURE;
    }
  }
  return ret;
}