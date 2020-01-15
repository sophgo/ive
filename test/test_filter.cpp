#include "tpu/tpu_filter.hpp"

#include "kernel_generator.hpp"
#include "opencv2/opencv.hpp"

int main() {
  // Create instance
  bmctx_t ctx;
  bmk1880v2_context_t *bk_ctx;
  createHandle(&ctx, &bk_ctx);
  printf("BM Kernel init.\n");

  // Fetch image information
  cv::Mat img = cv::imread("cat.png", 0);
  cv::resize(img, img, cv::Size(640, 360));
  CviImg cvi_img(&ctx, img.channels(), img.rows, img.cols, FMT_U8);
  memcpy(cvi_img.GetVAddr(), img.data, img.channels() * img.cols * img.rows);

  u32 kernel_size = 3;
  IveKernel kernel =
      createKernel(&ctx, img.channels(), kernel_size, kernel_size, IVE_KERNEL::GAUSSIAN);
  CviImg result(&ctx, img.channels(), img.rows, img.cols, FMT_U8);

  std::vector<CviImg> inputs, outputs;
  inputs.emplace_back(cvi_img);
  outputs.emplace_back(result);

  printf("Run TPU Filter.\n");
  // Run TPU Add
  IveTPUFilter tpu_filter;
  tpu_filter.init(&ctx, bk_ctx);
  tpu_filter.setKernel(kernel);
  tpu_filter.runSingleSizeKernel(&ctx, bk_ctx, inputs, &outputs);

  // write result to disk
  cv::Mat img1(img.rows, img.cols, CV_8UC1, result.GetVAddr());
  cv::imwrite("test_filter.png", img1);

  // Free memory, instance
  cvi_img.Free(&ctx);
  kernel.img.Free(&ctx);
  result.Free(&ctx);
  destroyHandle(&ctx);
  return 0;
}
