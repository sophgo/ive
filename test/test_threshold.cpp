#include "tpu/tpu_threshold.hpp"

#include "opencv2/opencv.hpp"

int main() {
  // Create instance
  bmctx_t ctx;
  bmk1880v2_context_t *bk_ctx;
  createHandle(&ctx, &bk_ctx);
  printf("BM Kernel init.\n");

  // Fetch image information
  cv::Mat img = cv::imread("cat.png", 0);
  CviImg cvi_img(&ctx, img.channels(), img.rows, img.cols, FMT_U8);
  memcpy(cvi_img.GetVAddr(), img.data, img.channels() * img.cols * img.rows);
  cvi_img.Flush(&ctx);
  CviImg result(&ctx, img.channels(), img.rows, img.cols, FMT_U8);

  std::vector<CviImg> inputs, outputs;
  inputs.emplace_back(cvi_img);
  outputs.emplace_back(result);

  printf("Run TPU Threshold.\n");
  // Run TPU Add
  IveTPUThreshold tpu_threshold;
  tpu_threshold.init(&ctx, bk_ctx);
  tpu_threshold.setThreshold(170);
  tpu_threshold.runSingleSizeKernel(&ctx, bk_ctx, inputs, &outputs);

  // write result to disk
  result.Invld(&ctx);
  cv::Mat img1(img.rows, img.cols, CV_8UC1, result.GetVAddr());
  cv::imwrite("test_threshold.png", img1);

  // Free memory, instance
  cvi_img.Free(&ctx);
  result.Free(&ctx);
  destroyHandle(&ctx);
  return 0;
}