#include "ive.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
  if (argc != 1) {
    printf("Incorrect loop value. Usage: %s\n", argv[0]);
    return CVI_FAILURE;
  }
  // Redirect logging to file.
  CVI_SYS_LOGGING(argv[0]);
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  // Create src image.
  const CVI_U16 width = 16;
  const CVI_U16 height = 10;
  IVE_SRC_IMAGE_S src;
  CVI_IVE_CreateImage(handle, &src, IVE_IMAGE_TYPE_U8C1, width, height);
  // Use rand to generate input data.
  srand(time(NULL));
  CVI_U32 srcLength = (CVI_U32)(width * height);
  for (CVI_U32 i = 0; i < srcLength; i++) {
    src.pu8VirAddr[0][i] = rand() % 256;
  }
  // Flush data to DRAM before TPU use.
  CVI_IVE_BufFlush(handle, &src);

  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, width, height);

  // Generate table for CVI_IVE_Map.
  IVE_DST_MEM_INFO_S dstTbl;
  CVI_U32 dstTblByteSize = 256;
  CVI_IVE_CreateMemInfo(handle, &dstTbl, dstTblByteSize);
  for (CVI_U32 i = 0; i < dstTblByteSize; i++) {
    dstTbl.pu8VirAddr[i] = dstTblByteSize - 1 - i;
  }

  printf("Run TPU Map.\n");
  int ret = CVI_IVE_Map(handle, &src, &dstTbl, &dst, 0);

  // Refresh CPU cache before CPU use.
  CVI_IVE_BufRequest(handle, &dst);
  // Print the array.
  printf("Input data:\n");
  for (CVI_U16 i = 0; i < height; i++) {
    for (CVI_U16 j = 0; j < width; j++) {
      printf("%3u ", src.pu8VirAddr[0][i * width + j]);
    }
    printf("\n");
  }
  // Print the array.
  printf("Table:\n");
  for (CVI_U32 i = 0; i < dstTblByteSize; i++) {
    printf("%3u ", dstTbl.pu8VirAddr[i]);
  }
  printf("\n");
  // Print the array.
  printf("Output result:\n");
  for (CVI_U16 i = 0; i < height; i++) {
    for (CVI_U16 j = 0; j < width; j++) {
      printf("%3u ", dst.pu8VirAddr[0][i * width + j]);
    }
    printf("\n");
  }

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dst);
  CVI_SYS_FreeM(handle, &dstTbl);
  CVI_IVE_DestroyHandle(handle);

  return ret;
}