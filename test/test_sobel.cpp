#include "tpu/tpu_sobel.hpp"

#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "kernel_generator.hpp"
#include "opencv2/opencv.hpp"

#include <glog/logging.h>

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  // Create instance
  bmctx_t ctx;
  bmk1880v2_context_t *bk_ctx;
  createHandle(&ctx, &bk_ctx);
  printf("BM Kernel init.\n");

  // Fetch image information
  cv::Mat img = cv::imread("cat.png", 0);
  // cv::resize(img, img, cv::Size(640, 360));
  CviImg cvi_img(&ctx, img.channels(), img.rows, img.cols, FMT_U8);
  memcpy(cvi_img.GetVAddr(), img.data, img.channels() * img.cols * img.rows);
  cvi_img.Flush(&ctx);
  u32 kernel_size = 3;
  IveKernel kernel_x =
      createKernel(&ctx, img.channels(), kernel_size, kernel_size, IVE_KERNEL::SOBEL_X);
  IveKernel kernel_y =
      createKernel(&ctx, img.channels(), kernel_size, kernel_size, IVE_KERNEL::SOBEL_Y);
  CviImg result(&ctx, img.channels(), img.rows, img.cols, FMT_BF16);

  std::vector<CviImg> inputs, outputs;
  inputs.emplace_back(cvi_img);
  outputs.emplace_back(result);

  printf("Run TPU Sobel.\n");
  // Run TPU Add
  IveTPUSobel tpu_sobel;
  tpu_sobel.init(&ctx, bk_ctx);
  tpu_sobel.setKernel(kernel_x, kernel_y);
  tpu_sobel.runSingleSizeKernel(&ctx, bk_ctx, inputs, &outputs);

  // write result to disk
  result.Invld(&ctx);
  cv::Mat img1(img.rows, img.cols, CV_32FC1);
  float *img1_ptr = (float *)img1.data;
  s16 *result_ptr = (s16 *)result.GetVAddr();
  for (size_t i = 0; i < (size_t)(img.channels() * img.rows * img.cols); i++) {
    img1_ptr[i] = convert_bf16_fp32(result_ptr[i]);
  }
  cv::Mat img1_u8;
  cv::normalize(img1, img1_u8, 255, 0, cv::NORM_MINMAX);
  cv::imwrite("test_sobel.png", img1_u8);

  // Free memory, instance
  cvi_img.Free(&ctx);
  kernel_x.img.Free(&ctx);
  kernel_y.img.Free(&ctx);
  result.Free(&ctx);
  destroyHandle(&ctx);
  return 0;
}
