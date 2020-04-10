#include "bmkernel/bm_kernel.h"

#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "ive.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __ARM_ARCH
#include "arm_neon.h"
#endif
#define CELL_SIZE 8
#define BLOCK_SIZE 2
#define STEP_X 1
#define STEP_Y 1
#define BIN_NUM 9

int cpu_ref(const int channels, IVE_SRC_IMAGE_S *src, IVE_DST_IMAGE_S *dstH, IVE_DST_IMAGE_S *dstV,
            IVE_DST_IMAGE_S *dstAng);

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <fullpath.txt> <number> <out_fea.txt>\n", argv[0]);
    return CVI_FAILURE;
  }
  CVI_SYS_LOGGING(argv[0]);
  const char *img_path_file = argv[1];
  const char *fea_path_file = argv[3];
  size_t total_run = atoi(argv[2]);  // total number of images

#if 0
  if (argc != 3) {
    printf("Incorrect loop value. Usage: %s <file name> <loop in value (1-1000)>\n", argv[0]);
    return CVI_FAILURE;
  }
  CVI_SYS_LOGGING(argv[0]);
  const char *filename = argv[1];
  size_t total_run = atoi(argv[2]);
//#endif
  printf("Loop value: %lu\n", total_run);
  if (total_run > 1000 || total_run == 0) {
    printf("Incorrect loop value. Usage: %s <file name> <loop in value (1-1000)>\n", argv[0]);
    return CVI_FAILURE;
  }
#endif
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");
  IVE_DST_IMAGE_S dstH, dstV;
  IVE_DST_IMAGE_S dstH_u8, dstV_u8;
  IVE_DST_IMAGE_S dstMag, dstAng;
  IVE_DST_IMAGE_S dstAng_u8;
  IVE_DST_MEM_INFO_S dstHist;
  CVI_U32 dstHistSize = 0;
  IVE_HOG_CTRL_S pstHogCtrl;
  int ret = 1;

  bool binit = false;

  char image_full_path[256];
  FILE *fpPath = fopen(img_path_file, "r");

  FILE *fpFea = fopen(fea_path_file, "w");
  for (int i = 0; i < total_run; i++) {
    fscanf(fpPath, "%s", image_full_path);
    printf("[0]: %s\n", image_full_path);
    // Fetch image information
    IVE_IMAGE_S src = CVI_IVE_ReadImage(handle, image_full_path, IVE_IMAGE_TYPE_U8C1);
    // int nChannels = 1;
    int width = src.u16Width;
    int height = src.u16Height;
    if (!binit) {
      // IVE_DST_IMAGE_S dstH, dstV;
      CVI_IVE_CreateImage(handle, &dstV, IVE_IMAGE_TYPE_BF16C1, width, height);
      CVI_IVE_CreateImage(handle, &dstH, IVE_IMAGE_TYPE_BF16C1, width, height);

      // IVE_DST_IMAGE_S dstH_u8, dstV_u8;
      CVI_IVE_CreateImage(handle, &dstH_u8, IVE_IMAGE_TYPE_U8C1, width, height);
      CVI_IVE_CreateImage(handle, &dstV_u8, IVE_IMAGE_TYPE_U8C1, width, height);

      // IVE_DST_IMAGE_S dstAng;
      CVI_IVE_CreateImage(handle, &dstMag, IVE_IMAGE_TYPE_BF16C1, width, height);
      CVI_IVE_CreateImage(handle, &dstAng, IVE_IMAGE_TYPE_BF16C1, width, height);

      // IVE_DST_IMAGE_S dstAng_u8;
      CVI_IVE_CreateImage(handle, &dstAng_u8, IVE_IMAGE_TYPE_U8C1, width, height);

      // IVE_DST_MEM_INFO_S dstHist;
      // CVI_U32 dstHistSize = 0;
      CVI_IVE_GET_HOG_SIZE(dstAng.u16Width, dstAng.u16Height, BIN_NUM, CELL_SIZE, BLOCK_SIZE,
                           STEP_X, STEP_Y, &dstHistSize);
      CVI_IVE_CreateMemInfo(handle, &dstHist, dstHistSize);

      pstHogCtrl.u8BinSize = BIN_NUM;
      pstHogCtrl.u32CellSize = CELL_SIZE;
      pstHogCtrl.u16BlkSize = BLOCK_SIZE;
      pstHogCtrl.u16BlkStepX = STEP_X;
      pstHogCtrl.u16BlkStepY = STEP_Y;
      binit = true;
    }
    printf("Run TPU HOG. len: %d\n", dstHistSize);
    // IVE_HOG_CTRL_S pstHogCtrl;

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    // for (size_t i = 0; i < total_run; i++)
    {
      CVI_IVE_HOG(handle, &src, &dstH, &dstV, &dstMag, &dstAng, &dstHist, &pstHogCtrl, 0);
      float *ptr = (float *)dstHist.pu8VirAddr;
      int reald = dstHistSize / sizeof(u32);
      for (int j = 0; j < reald; j++) {
        // printf("%d %d\n", j, ptr[j]  );
        fprintf(fpFea, "%f ", ptr[j]);
      }
      fprintf(fpFea, "\n");
    }
    gettimeofday(&t1, NULL);
    unsigned long elapsed_tpu =
        ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec) / total_run;

    printf("Normalize result to 0-255.\n");
    IVE_ITC_CRTL_S iveItcCtrl;
    iveItcCtrl.enType = IVE_ITC_NORMALIZE;
    CVI_IVE_ImageTypeConvert(handle, &dstV, &dstV_u8, &iveItcCtrl, 0);
    CVI_IVE_ImageTypeConvert(handle, &dstH, &dstH_u8, &iveItcCtrl, 0);
    CVI_IVE_ImageTypeConvert(handle, &dstAng, &dstAng_u8, &iveItcCtrl, 0);

    CVI_IVE_BufRequest(handle, &src);
    CVI_IVE_BufRequest(handle, &dstH);
    CVI_IVE_BufRequest(handle, &dstV);
    CVI_IVE_BufRequest(handle, &dstAng);
    // ret = cpu_ref(nChannels, &src, &dstH, &dstV, &dstAng);
    if (total_run == 1) {
      // write result to disk
      printf("Save to image.\n");
      CVI_IVE_WriteImage(handle, "test_sobelV_c.png", &dstV_u8);
      CVI_IVE_WriteImage(handle, "test_sobelH_c.png", &dstH_u8);
      CVI_IVE_WriteImage(handle, "test_ang_c.png", &dstAng_u8);
      printf("Output HOG feature.\n");
      /*
      u32 blkSize = BLOCK_SIZE * BLOCK_SIZE * BIN_NUM;
      u32 blkNum = dstHistSize / sizeof(float) / blkSize;
      for (size_t i = 0; i < blkNum; i++) {
        //printf("\n");
        for (size_t j = 0; j < blkSize; j++) {
          printf("%3f ", ((float *)dstHist.pu8VirAddr)[i + j]);
        }
      }
      printf("\n");*/

    } else {
      printf("OOO %-10s %10lu %10s %10s\n", "HOG", elapsed_tpu, "NA", "NA");
    }

    CVI_SYS_FreeI(handle, &src);
  }
  fclose(fpFea);
  fclose(fpPath);
  // Free memory, instance
  // CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dstH);
  CVI_SYS_FreeI(handle, &dstH_u8);
  CVI_SYS_FreeI(handle, &dstV);
  CVI_SYS_FreeI(handle, &dstV_u8);
  CVI_SYS_FreeI(handle, &dstAng);
  CVI_SYS_FreeI(handle, &dstAng_u8);
  CVI_SYS_FreeM(handle, &dstHist);
  CVI_IVE_DestroyHandle(handle);

  return ret;
}

int cpu_ref(const int channels, IVE_SRC_IMAGE_S *src, IVE_DST_IMAGE_S *dstH, IVE_DST_IMAGE_S *dstV,
            IVE_DST_IMAGE_S *dstAng) {
  int ret = CVI_SUCCESS;
  u16 *dstH_ptr = (u16 *)dstH->pu8VirAddr[0];
  u16 *dstV_ptr = (u16 *)dstV->pu8VirAddr[0];
  u16 *dstAng_ptr = (u16 *)dstAng->pu8VirAddr[0];
  float mul_val = 180.f / M_PI;
  float ang_abs_limit = 1;

  printf("Check Ang:\n");
  for (size_t i = 0; i < channels * src->u16Width * src->u16Height; i++) {
    float dstH_f = convert_bf16_fp32(dstH_ptr[i]);
    float dstV_f = convert_bf16_fp32(dstV_ptr[i]);
    float dstAng_f = convert_bf16_fp32(dstAng_ptr[i]);
    float atan2_res = (float)atan2(dstV_f, dstH_f) * mul_val;
    float error = fabs(atan2_res - dstAng_f);
    if (error > ang_abs_limit) {
      // printf("[%lu] atan2( %f, %f) = TPU %f, CPU %f. eplison = %f\n", i, dstV_f, dstH_f,
      // dstAng_f,
      //       atan2_res, error);
      ret = CVI_FAILURE;
    }
  }
  return ret;
}