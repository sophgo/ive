#include "tpu/tpu_add.hpp"

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
  memset(black_img.GetVAddr(), 255, img.channels() * img.cols * img.rows);
  for (int j = img.rows / 10; j < img.rows * 9 / 10; j++) {
    for (int i = img.cols / 10; i < img.cols * 9 / 10; i++) {
      black_img.GetVAddr()[i + j * img.cols] = 0;
    }
  }
  CviImg result(&ctx, img.channels(), img.rows, img.cols, FMT_U8);

  std::vector<CviImg> inputs, outputs;
  inputs.emplace_back(cvi_img);
  inputs.emplace_back(black_img);
  outputs.emplace_back(result);

  printf("Run TPU Add.\n");
  // Run TPU Add
  IveTPUAdd tpu_add;
  tpu_add.init(&ctx, bk_ctx);
  tpu_add.runSingleSizeKernel(&ctx, bk_ctx, inputs, &outputs);

  // write result to disk
  cv::Mat img1(img.rows, img.cols, CV_8UC1, result.GetVAddr());
  cv::imwrite("test_add.png", img1);

  // Free memory, instance
  cvi_img.Free(&ctx);
  black_img.Free(&ctx);
  result.Free(&ctx);
  destroyHandle(&ctx);
  return 0;
}
