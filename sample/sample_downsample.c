#include "ive.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CELL_SZ 4

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <file name>\n", argv[0]);
    return CVI_FAILURE;
  }
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  const char *file_name = argv[1];
  IVE_IMAGE_S src = CVI_IVE_ReadImage(handle, file_name, IVE_IMAGE_TYPE_U8C1);

  IVE_DOWNSAMPLE_CTRL_S ive_ds_Ctrl;
  ive_ds_Ctrl.u8KnerlSize = CELL_SZ;
  // The size of the output must meet the requirement of
  int res_w = src.u16Width / CELL_SZ;
  int res_h = src.u16Height / CELL_SZ;
  // IVE_DST_IMAGE_S dst;
  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, res_w, res_h);

  printf("Run TPU DownSample.\n");
  int ret = CVI_IVE_DOWNSAMPLE(handle, &src, &dst, &ive_ds_Ctrl, 0);

  CVI_IVE_BufRequest(handle, &dst);

  CVI_IVE_WriteImage(handle, "test.png", &dst);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dst);
  CVI_IVE_DestroyHandle(handle);

  return ret;
}