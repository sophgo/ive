#include "ive.h"

#include <stdio.h>
#include <string.h>

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
  memset(src2.pu8VirAddr[0], 255, nChannels * width * height);
  for (int j = height / 10; j < height * 9 / 10; j++) {
    for (int i = width / 10; i < width * 9 / 10; i++) {
      src2.pu8VirAddr[0][i + j * width] = 0;
    }
  }

  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, width, height);

  printf("Run TPU Sub.\n");
  IVE_SUB_CTRL_S iveSubCtrl;
  iveSubCtrl.enMode = IVE_SUB_MODE_ABS;
  CVI_IVE_Sub(handle, &src1, &src2, &dst, &iveSubCtrl, 0);

  // write result to disk
  printf("Save to image.\n");
  CVI_IVE_WriteImage("test_sub_c.png", &dst);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src1);
  CVI_SYS_FreeI(handle, &src2);
  CVI_SYS_FreeI(handle, &dst);
  CVI_IVE_DestroyHandle(handle);

  return 0;
}
