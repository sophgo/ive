#include "ive.h"

#include <stdio.h>
#include <string.h>

int cpu_ref(const int channels, IVE_SRC_IMAGE_S *src1, IVE_SRC_IMAGE_S *src2, IVE_DST_IMAGE_S *dst);

int main(int argc, char **argv) {
  CVI_SYS_LOGGING(argv[0]);
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  // Fetch image information
  IVE_IMAGE_S src1 = CVI_IVE_ReadImage(handle, "cat.png", IVE_IMAGE_TYPE_U8C1);
  int nChannels = 1;
  int width = src1.u16Width;
  int height = src1.u16Height;

  IVE_SRC_IMAGE_S src2;
  CVI_IVE_CreateImage(handle, &src2, IVE_IMAGE_TYPE_U8C1, width, height);
  memset(src2.pu8VirAddr[0], 0, nChannels * width * height);
  for (int j = height / 10; j < height * 9 / 10; j++) {
    for (int i = width / 10; i < width * 9 / 10; i++) {
      src2.pu8VirAddr[0][i + j * width] = 255;
    }
  }

  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, width, height);

  printf("Run TPU And.\n");
  CVI_IVE_And(handle, &src1, &src2, &dst, 0);

  int ret = cpu_ref(nChannels, &src1, &src2, &dst);

  // write result to disk
  printf("Save to image.\n");
  CVI_IVE_WriteImage("test_and_c.png", &dst);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src1);
  CVI_SYS_FreeI(handle, &src2);
  CVI_SYS_FreeI(handle, &dst);
  CVI_IVE_DestroyHandle(handle);

  return ret;
}

int cpu_ref(const int channels, IVE_SRC_IMAGE_S *src1, IVE_SRC_IMAGE_S *src2,
            IVE_DST_IMAGE_S *dst) {
  int ret = CVI_SUCCESS;
  CVI_U8 *src1_ptr = src1->pu8VirAddr[0];
  CVI_U8 *src2_ptr = src2->pu8VirAddr[0];
  CVI_U8 *dst_ptr = dst->pu8VirAddr[0];
  for (size_t i = 0; i < channels * src1->u16Width * src1->u16Height; i++) {
    int res = src1_ptr[i] & src2_ptr[i];
    if (res != dst_ptr[i]) {
      printf("[%lu] %d & %d = TPU %d, CPU %d\n", i, src1_ptr[i], src2_ptr[i], dst_ptr[i],
             (int)src1_ptr[i] + src2_ptr[i]);
      ret = CVI_FAILURE;
      break;
    }
  }
  return ret;
}