#include "tpu/tpu_sub.hpp"

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
  CviImg black_img(&ctx, img.channels(), img.rows, img.cols, FMT_U8);
  memset(black_img.GetVAddr(), 0, img.channels() * img.cols * img.rows);
  cvi_img.Flush(&ctx);
  black_img.Flush(&ctx);
  CviImg result(&ctx, img.channels(), img.rows, img.cols, FMT_U8);

  std::vector<CviImg> inputs, outputs;
  inputs.emplace_back(cvi_img);
  inputs.emplace_back(black_img);
  outputs.emplace_back(result);

  printf("Run TPU Sub.\n");
  // Run TPU Add
  IveTPUSub tpu_sub;
  tpu_sub.init(&ctx, bk_ctx);
  tpu_sub.runSingleSizeKernel(&ctx, bk_ctx, inputs, &outputs);

  // write result to disk
  result.Invld(&ctx);
  cv::Mat img1(img.rows, img.cols, CV_8UC1, result.GetVAddr());
  cv::imwrite("test_sub.png", img1);

  // Free memory, instance
  cvi_img.Free(&ctx);
  black_img.Free(&ctx);
  result.Free(&ctx);
  destroyHandle(&ctx);
  return 0;
}
