#include "bmkernel/bm_kernel.h"
#include "ive.h"

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#ifdef __ARM_ARCH
#include "arm_neon.h"
#endif

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
  memset(src2.pu8VirAddr[0], 255, nChannels * width * height);
  for (int j = height / 10; j < height * 9 / 10; j++) {
    for (int i = width / 10; i < width * 9 / 10; i++) {
      src2.pu8VirAddr[0][i + j * width] = 0;
    }
  }
  CVI_IVE_BufFlush(handle, &src2);

  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, width, height);

  printf("Run TPU Add.\n");
  IVE_ADD_CTRL_S iveAddCtrl;
  CVI_U16 res = convert_fp32_bf16(1.f);
  iveAddCtrl.u0q16X = res;
  iveAddCtrl.u0q16Y = res;
  unsigned long long total_t = 0;
  struct timeval t0, t1;
  for (size_t i = 0; i < 1; i++) {
    gettimeofday(&t0, NULL);
    CVI_IVE_Add(handle, &src1, &src2, &dst, &iveAddCtrl, 0);
    gettimeofday(&t1, NULL);
    unsigned long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("[%lu] time elapsed %lu\n", i, elapsed);
    total_t += elapsed;
  }
  printf("total time %llu\n", total_t);
  CVI_IVE_BufRequest(handle, &src1);
  CVI_IVE_BufRequest(handle, &src2);
  CVI_IVE_BufRequest(handle, &dst);
  int ret = cpu_ref(nChannels, &src1, &src2, &dst);
#ifdef __ARM_ARCH
  uint8_t *ptr1 = src1.pu8VirAddr[0];
  uint8_t *ptr2 = src2.pu8VirAddr[0];
  uint8_t *ptr3 = dst.pu8VirAddr[0];
  gettimeofday(&t0, NULL);
  for (size_t i = 0; i < width * height / 16; i++) {
    uint8x16_t val = vld1q_u8(ptr1);
    uint8x16_t val2 = vld1q_u8(ptr2);
    uint8x16_t result = vqaddq_u8(val, val2);
    vst1q_u8(ptr3, result);
    ptr1 += 16;
    ptr2 += 16;
    ptr3 += 16;
  }
  gettimeofday(&t1, NULL);
  unsigned long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
  printf("cpu time neon elapsed %lu\n", elapsed);
#endif

  // write result to disk
  printf("Save to image.\n");
  CVI_IVE_WriteImage(handle, "test_add_c.png", &dst);

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
    int res = src1_ptr[i] + src2_ptr[i];
    res = res > 255 ? 255 : res;
    if (res != dst_ptr[i]) {
      printf("[%lu] %d + %d = TPU %d, CPU %d\n", i, src1_ptr[i], src2_ptr[i], dst_ptr[i], (int)res);
      ret = CVI_FAILURE;
      break;
    }
  }
  return ret;
}