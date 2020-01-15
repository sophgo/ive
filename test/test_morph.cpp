#include "tpu/tpu_filter.hpp"
#include "tpu/tpu_morph.hpp"
#include "tpu/tpu_threshold.hpp"

#include "kernel_generator.hpp"
#include "opencv2/opencv.hpp"

#include <glog/logging.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // Create instance
  bmctx_t ctx;
  bmk1880v2_context_t* bk_ctx;
  createHandle(&ctx, &bk_ctx);
  printf("BM Kernel init.\n");

  // Fetch image information
  cv::Mat img = cv::imread("cat.png", 0);
  CviImg cvi_img(&ctx, img.channels(), img.rows, img.cols, FMT_U8);
  memcpy(cvi_img.GetVAddr(), img.data, img.channels() * img.cols * img.rows);
  CviImg result(&ctx, img.channels(), img.rows, img.cols, FMT_U8);

  std::vector<CviImg> inputs, outputs;
  inputs.emplace_back(cvi_img);
  outputs.emplace_back(result);

  printf("Run TPU Threshold + Morph.\n");
  // Run TPU Add
  IveTPUThreshold tpu_threshold;
  tpu_threshold.init(&ctx, bk_ctx);
  tpu_threshold.setThreshold(170);
  tpu_threshold.runSingleSizeKernel(&ctx, bk_ctx, inputs, &outputs);

  u32 kernel_size = 5;
  IveKernel kernel =
      createKernel(&ctx, img.channels(), kernel_size, kernel_size, IVE_KERNEL::MORPH_CROSS);
  CviImg result2(&ctx, img.channels(), img.rows, img.cols, FMT_U8);

  inputs.clear();
  outputs.clear();
  inputs.emplace_back(result);
  outputs.emplace_back(result2);
  IveTPUErode tpu_erode;
  tpu_erode.setKernel(kernel);
  tpu_erode.init(&ctx, bk_ctx);
  tpu_erode.runSingleSizeKernel(&ctx, bk_ctx, inputs, &outputs);

  // write result to disk
  cv::Mat img2(img.rows, img.cols, CV_8UC1, result2.GetVAddr());
  cv::imwrite("test_erode.png", img2);

  IveTPUFilter tpu_filter;
  tpu_filter.setKernel(kernel);
  tpu_filter.init(&ctx, bk_ctx);
  tpu_filter.runSingleSizeKernel(&ctx, bk_ctx, inputs, &outputs);

  // write result to disk
  img2.data = result2.GetVAddr();
  // cv::Mat img3(img.rows, img.cols, CV_8UC1, result3.GetVAddr());
  cv::imwrite("test_dilate.png", img2);

  // write result to disk
  cv::Mat img1(img.rows, img.cols, CV_8UC1, result.GetVAddr());
  cv::imwrite("test_morph_raw.png", img1);
  // Free memory, instance
  cvi_img.Free(&ctx);
  result.Free(&ctx);
  kernel.img.Free(&ctx);
  result2.Free(&ctx);
  destroyHandle(&ctx);
  return 0;
}