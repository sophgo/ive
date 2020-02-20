#include "ive.h"

#include <stdio.h>
#include "opencv2/opencv.hpp"

#define CAM_CHANNEL 1
#define CAM_WIDTH 1280
#define CAM_HEIGHT 720
#define CAM_TOTAL_SZ (CAM_CHANNEL * CAM_WIDTH * CAM_HEIGHT)
#define TOTAL_FRAME 10

int main(int argc, char **argv) {
  CVI_SYS_LOGGING(argv[0]);

  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  // Fetch image information
  cv::VideoCapture cap;
  cap.open(0);
  if (!cap.isOpened()) {
    printf("Failed to open camera.\n");
    return CVI_FAILURE;
  }
  cap.set(CV_CAP_PROP_FRAME_WIDTH, CAM_WIDTH);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT);

  // Create image buffers
  IVE_SRC_IMAGE_S src[2], tmp, andframe[2], dst;
  CVI_IVE_CreateImage(handle, &src[0], IVE_IMAGE_TYPE_U8C1, CAM_WIDTH, CAM_HEIGHT);
  CVI_IVE_CreateImage(handle, &src[1], IVE_IMAGE_TYPE_U8C1, CAM_WIDTH, CAM_HEIGHT);
  CVI_IVE_CreateImage(handle, &tmp, IVE_IMAGE_TYPE_U8C1, CAM_WIDTH, CAM_HEIGHT);
  CVI_IVE_CreateImage(handle, &andframe[0], IVE_IMAGE_TYPE_U8C1, CAM_WIDTH, CAM_HEIGHT);
  CVI_IVE_CreateImage(handle, &andframe[1], IVE_IMAGE_TYPE_U8C1, CAM_WIDTH, CAM_HEIGHT);
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, CAM_WIDTH, CAM_HEIGHT);

  cv::Mat frame, frame_gray;
  int count = 0;
  for (size_t i = 0; i < TOTAL_FRAME; i++) {
    printf("Frame no.%lu\n", i);
    // Get frame from VideoCapture
    cap >> frame;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    // Copy to input buffer
    memcpy(src[count].pu8VirAddr[0], frame_gray.data, CAM_TOTAL_SZ);
    if (i > 0) {
      // Sub - threshold - dilate
      IVE_SUB_CTRL_S iveSubCtrl;
      iveSubCtrl.enMode = IVE_SUB_MODE_BUTT;
      CVI_IVE_Sub(handle, &src[count], &src[1 - count], &tmp, &iveSubCtrl, 0);

      IVE_THRESH_CTRL_S iveTshCtrl;
      iveTshCtrl.enMode = IVE_THRESH_MODE_BINARY;
      iveTshCtrl.u8MinVal = 0;
      iveTshCtrl.u8MaxVal = 255;
      iveTshCtrl.u8LowThr = 35;
      CVI_IVE_Thresh(handle, &tmp, &tmp, &iveTshCtrl, 0);

      IVE_DILATE_CTRL_S stDilateCtrl;
      memset(stDilateCtrl.au8Mask, 1, 25);
      CVI_IVE_Dilate(handle, &tmp, &andframe[count], &stDilateCtrl, 0);

      if (i > 1) {
        // And two dilated images
        CVI_IVE_And(handle, &andframe[count], &andframe[1 - count], &dst, 0);
      }
    }
    count = 1 - count;
  }

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src[0]);
  CVI_SYS_FreeI(handle, &src[1]);
  CVI_SYS_FreeI(handle, &tmp);
  CVI_SYS_FreeI(handle, &andframe[0]);
  CVI_SYS_FreeI(handle, &andframe[1]);
  CVI_SYS_FreeI(handle, &dst);
  CVI_IVE_DestroyHandle(handle);

  return CVI_SUCCESS;
}
