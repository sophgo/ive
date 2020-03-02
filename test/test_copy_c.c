#include "ive.h"

#include <stdio.h>

int cpu_ref(const int channels, const int x_sz, const int y_sz, IVE_SRC_IMAGE_S *src,
            IVE_DST_IMAGE_S *dst_copy, IVE_DST_IMAGE_S *dst_interval);

int main(int argc, char **argv) {
  CVI_SYS_LOGGING(argv[0]);
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  // Fetch image information
  IVE_IMAGE_S src = CVI_IVE_ReadImage(handle, "cat.png", IVE_IMAGE_TYPE_U8C1);
  int nChannels = 1;
  int width = src.u16Width;
  int height = src.u16Height;

  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, width, height);

  printf("Run TPU Direct Copy.\n");
  IVE_DMA_CTRL_S iveDmaCtrl;
  iveDmaCtrl.enMode = IVE_DMA_MODE_DIRECT_COPY;
  CVI_IVE_DMA(handle, &src, &dst, &iveDmaCtrl, 0);

  IVE_DST_IMAGE_S dst2;
  CVI_IVE_CreateImage(handle, &dst2, IVE_IMAGE_TYPE_U8C1, width * 2, height * 2);

  printf("Run TPU Interval Copy.\n");
  iveDmaCtrl.enMode = IVE_DMA_MODE_INTERVAL_COPY;
  iveDmaCtrl.u8HorSegSize = 2;
  iveDmaCtrl.u8VerSegRows = 2;
  CVI_IVE_DMA(handle, &src, &dst2, &iveDmaCtrl, 0);

  IVE_DST_IMAGE_S src_crop;
  CVI_IVE_SubImage(handle, &src, &src_crop, 100, 100, 1380, 820);
  IVE_DST_IMAGE_S dst3;
  CVI_IVE_CreateImage(handle, &dst3, IVE_IMAGE_TYPE_U8C1, 1280, 720);

  printf("Run TPU Sub Direct Copy.\n");
  iveDmaCtrl.enMode = IVE_DMA_MODE_DIRECT_COPY;
  CVI_IVE_DMA(handle, &src_crop, &dst3, &iveDmaCtrl, 0);

  CVI_IVE_BufRequest(handle, &src);
  CVI_IVE_BufRequest(handle, &dst);
  CVI_IVE_BufRequest(handle, &dst2);
  CVI_IVE_BufRequest(handle, &dst3);
  int ret = cpu_ref(nChannels, iveDmaCtrl.u8HorSegSize, iveDmaCtrl.u8VerSegRows, &src, &dst, &dst2);

  // write result to disk
  printf("Save to image.\n");
  CVI_IVE_WriteImage(handle, "test_dcopy_c.png", &dst);
  CVI_IVE_WriteImage(handle, "test_icopy_c.png", &dst2);
  CVI_IVE_WriteImage(handle, "test_scopy_c.png", &dst3);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dst);
  CVI_SYS_FreeI(handle, &dst2);
  CVI_SYS_FreeI(handle, &dst3);
  CVI_IVE_DestroyHandle(handle);

  return ret;
}

int cpu_ref(const int channels, const int x_sz, const int y_sz, IVE_SRC_IMAGE_S *src,
            IVE_DST_IMAGE_S *dst_copy, IVE_DST_IMAGE_S *dst_interval) {
  int ret = CVI_SUCCESS;
  for (size_t i = 0; i < channels * src->u16Width * src->u16Height; i++) {
    if (src->pu8VirAddr[0][i] != dst_copy->pu8VirAddr[0][i]) {
      printf("[%lu] original %d copied %d", i, src->pu8VirAddr[0][i], dst_copy->pu8VirAddr[0][i]);
      ret = CVI_FAILURE;
      break;
    }
  }
  for (size_t k = 0; k < channels; k++) {
    for (size_t i = 0; i < 6; i++) {
      for (size_t j = 0; j < dst_interval->u16Width; j++) {
        char src_val = src->pu8VirAddr[0][(j / x_sz) + (i / y_sz) * src->u16Stride[0] +
                                          k * src->u16Stride[0] * src->u16Height];
        char dst_val =
            dst_interval->pu8VirAddr[0][j + i * dst_interval->u16Stride[0] +
                                        k * dst_interval->u16Stride[0] * dst_interval->u16Height];
        if (i % y_sz != 0 || j % x_sz != 0) {
          src_val = 0;
        }
        if (src_val != dst_val) {
          printf("[%lu][%lu][%lu] original %d copied %d\n", k, i, j, src_val, dst_val);
          ret = CVI_FAILURE;
          break;
        }
      }
    }
  }
  return ret;
}