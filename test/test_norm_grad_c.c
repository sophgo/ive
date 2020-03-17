#include "bmkernel/bm_kernel.h"
#include "ive.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

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
  int width = src.u16Width;
  int height = src.u16Height;

  IVE_DST_IMAGE_S dstH_u8, dstV_u8, dstHV_u8;
  CVI_IVE_CreateImage(handle, &dstH_u8, IVE_IMAGE_TYPE_U8C1, width, height);
  CVI_IVE_CreateImage(handle, &dstV_u8, IVE_IMAGE_TYPE_U8C1, width, height);
  CVI_IVE_CreateImage(handle, &dstHV_u8, IVE_IMAGE_TYPE_U8C1, width, height);

  printf("Run norm gradient.\n");
  IVE_NORM_GRAD_CTRL_S pstNormGradCtrl;
  pstNormGradCtrl.enOutCtrl = IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER;
  struct timeval t0, t1;
  gettimeofday(&t0, NULL);
  for (size_t i = 0; i < total_run; i++) {
    CVI_IVE_NormGrad(handle, &src, &dstH_u8, &dstV_u8, &dstHV_u8, &pstNormGradCtrl, 0);
  }
  gettimeofday(&t1, NULL);
  unsigned long elapsed_tpu_hv =
      ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec) / total_run;
  printf("Run norm gradient combine.\n");
  pstNormGradCtrl.enOutCtrl = IVE_NORM_GRAD_OUT_CTRL_COMBINE;
  gettimeofday(&t0, NULL);
  for (size_t i = 0; i < total_run; i++) {
    CVI_IVE_NormGrad(handle, &src, &dstH_u8, &dstV_u8, &dstHV_u8, &pstNormGradCtrl, 0);
  }
  gettimeofday(&t1, NULL);
  unsigned long elapsed_tpu_combine =
      ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec) / total_run;

  if (total_run == 1) {
    printf("TPU norm grad avg time %lu\n", elapsed_tpu_hv);
    printf("TPU norm grad combine avg time %lu\n", elapsed_tpu_combine);
#ifdef __ARM_ARCH
    printf("CPU NEON time %s\n", "NA");
    printf("CPU time %s\n", "NA");
#endif
    // write result to disk
    printf("Save to image.\n");
    CVI_IVE_WriteImage(handle, "test_normV_c.png", &dstV_u8);
    CVI_IVE_WriteImage(handle, "test_normH_c.png", &dstH_u8);
    CVI_IVE_WriteImage(handle, "test_normHV_c.png", &dstHV_u8);
  }
#ifdef __ARM_ARCH
  else {
    printf("OOO %-10s %10lu %10s %10s\n", "NORM GRAD", elapsed_tpu_hv, "NA", "NA");
    printf("OOO %-10s %10lu %10s %10s\n", "NG + SQRT", elapsed_tpu_combine, "NA", "NA");
  }
#endif

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dstH_u8);
  CVI_SYS_FreeI(handle, &dstV_u8);
  CVI_SYS_FreeI(handle, &dstHV_u8);
  CVI_IVE_DestroyHandle(handle);

  return CVI_SUCCESS;
}