#ifndef _CVI_COMM_IVE_H_
#define _CVI_COMM_IVE_H_
#include "cvi_type.h"

typedef void *IVE_HANDLE;

#ifdef __cplusplus
extern "C" {
#endif
typedef struct CVI_IMG CVI_IMG_S;
#ifdef __cplusplus
}
#endif

typedef enum IVE_DMA_MODE {
  IVE_DMA_MODE_DIRECT_COPY = 0x0,
  IVE_DMA_MODE_INTERVAL_COPY = 0x1,
  IVE_DMA_MODE_SET_3BYTE = 0x2,
  IVE_DMA_MODE_SET_8BYTE = 0x3,
  IVE_DMA_MODE_BUTT
} IVE_DMA_MODE_E;

typedef struct IVE_DMA_CTRL {
  IVE_DMA_MODE_E enMode;
  CVI_U64 u64Val; /*Used in memset mode*/
  CVI_U8
  u8HorSegSize;      /*Used in interval-copy mode, every row was segmented by u8HorSegSize bytes,
                        restricted in values of 2,3,4,8,16*/
  CVI_U8 u8ElemSize; /*Used in interval-copy mode, the valid bytes copied in front of every segment
                        in a valid row, which 0<u8ElemSize<u8HorSegSize*/
  CVI_U8 u8VerSegRows; /*Used in interval-copy mode, copy one row in every u8VerSegRows*/
} IVE_DMA_CTRL_S;

typedef struct IVE_DATA {
  CVI_U32 u32PhyAddr; /*Physical address of the data*/
  CVI_U8 *pu8VirAddr;

  CVI_U16 u16Stride; /*2D data stride by byte*/
  CVI_U16 u16Width;  /*2D data width by byte*/
  CVI_U16 u16Height; /*2D data height*/

  CVI_U16 u16Reserved;
  CVI_IMG_S *tpu_block;
} IVE_DATA_S;

typedef IVE_DATA_S IVE_SRC_DATA_S;
typedef IVE_DATA_S IVE_DST_DATA_S;

typedef enum IVE_IMAGE_TYPE {
  IVE_IMAGE_TYPE_U8C1 = 0x0,
  IVE_IMAGE_TYPE_S8C1 = 0x1,

  IVE_IMAGE_TYPE_YUV420SP = 0x2, /*YUV420 SemiPlanar*/
  IVE_IMAGE_TYPE_YUV422SP = 0x3, /*YUV422 SemiPlanar*/
  IVE_IMAGE_TYPE_YUV420P = 0x4,  /*YUV420 Planar */
  IVE_IMAGE_TYPE_YUV422P = 0x5,  /*YUV422 planar */

  IVE_IMAGE_TYPE_S8C2_PACKAGE = 0x6,
  IVE_IMAGE_TYPE_S8C2_PLANAR = 0x7,

  IVE_IMAGE_TYPE_S16C1 = 0x8,
  IVE_IMAGE_TYPE_U16C1 = 0x9,

  IVE_IMAGE_TYPE_U8C3_PACKAGE = 0xa,
  IVE_IMAGE_TYPE_U8C3_PLANAR = 0xb,

  IVE_IMAGE_TYPE_S32C1 = 0xc,
  IVE_IMAGE_TYPE_U32C1 = 0xd,

  IVE_IMAGE_TYPE_S64C1 = 0xe,
  IVE_IMAGE_TYPE_U64C1 = 0xf,

  IVE_IMAGE_TYPE_BF16C1 = 0x10,
  IVE_IMAGE_TYPE_FP32C1 = 0x11,

  IVE_IMAGE_TYPE_BUTT

} IVE_IMAGE_TYPE_E;

typedef struct IVE_IMAGE {
  IVE_IMAGE_TYPE_E enType;

  CVI_U64 u64PhyAddr[3];
  CVI_U8 *pu8VirAddr[3];

  CVI_U16 u16Stride[3];
  CVI_U16 u16Width;
  CVI_U16 u16Height;

  CVI_U16 u16Reserved; /*Can be used such as elemSize*/
  CVI_IMG_S *tpu_block;
} IVE_IMAGE_S;

typedef IVE_IMAGE_S IVE_SRC_IMAGE_S;
typedef IVE_IMAGE_S IVE_DST_IMAGE_S;

typedef enum IVE_ITC_TYPE {
  IVE_ITC_SATURATE = 0x0,
  IVE_ITC_NORMALIZE = 0x1,
} IVE_ITC_TYPE_E;

typedef struct IVE_ITC_CRTL {
  IVE_ITC_TYPE_E enType;
} IVE_ITC_CRTL_S;

typedef struct IVE_ADD_CTRL_S {
  CVI_U0Q16 u0q16X; /*x of "xA+yB"*/
  CVI_U0Q16 u0q16Y; /*y of "xA+yB"*/
} IVE_ADD_CTRL_S;

typedef struct IVE_BLOCK_CTRL {
  CVI_FLOAT bin_num;
  CVI_U32 cell_size;
} IVE_BLOCK_CTRL_S;

typedef struct IVE_ELEMENT_STRUCTURE_CTRL {
  CVI_U8 au8Mask[25]; /*The template parameter value must be 0 or 255.*/
} IVE_ELEMENT_STRUCTURE_CTRL_S;

typedef IVE_ELEMENT_STRUCTURE_CTRL_S IVE_DILATE_CTRL_S;
typedef IVE_ELEMENT_STRUCTURE_CTRL_S IVE_ERODE_CTRL_S;

typedef struct IVE_FILTER_CTRL {
  CVI_S8 as8Mask[9];
  CVI_S8 u8Norm;
} IVE_FILTER_CTRL_S;

typedef enum IVE_NORM_GRAD_OUT_CTRL {
  IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER = 0x0,
  IVE_NORM_GRAD_OUT_CTRL_HOR = 0x1,
  IVE_NORM_GRAD_OUT_CTRL_VER = 0x2,
  IVE_NORM_GRAD_OUT_CTRL_COMBINE = 0x3,

  IVE_NORM_GRAD_OUT_CTRL_BUTT
} IVE_NORM_GRAD_OUT_CTRL_E;

/*
 *GradientFilter control parameters
 */
typedef struct IVE_NORM_GRAD_CTRL {
  IVE_NORM_GRAD_OUT_CTRL_E enOutCtrl;
  CVI_S8 as8Mask[25];
  CVI_U8 u8Norm;
} IVE_NORM_GRAD_CTRL_S;

typedef struct IVE_HOG_CTRL {
  CVI_U8 bin_num;
  CVI_U32 cell_size;
} IVE_HOG_CTRL_S;

typedef enum IVE_MAG_AND_ANG_OUT_CTRL {
  IVE_MAG_AND_ANG_OUT_CTRL_MAG = 0x0, /*Only the magnitude is output.*/
  IVE_MAG_AND_ANG_OUT_CTRL_ANG = 0x1,
  IVE_MAG_AND_ANG_OUT_CTRL_MAG_AND_ANG = 0x2, /*The magnitude and angle are output.*/
  IVE_MAG_AND_ANG_OUT_CTRL_BUTT
} IVE_MAG_AND_ANG_OUT_CTRL_E;

/*
 *Magnitude and angle control parameter
 */
typedef struct IVE_MAG_AND_ANG_CTRL {
  IVE_MAG_AND_ANG_OUT_CTRL_E enOutCtrl;
  CVI_U16 u16Thr;
  CVI_S8 as8Mask[25]; /*Template parameter.*/
  CVI_BOOL no_negative;
} IVE_MAG_AND_ANG_CTRL_S;

/*
 * Sad mode
 */
typedef enum IVE_SAD_MODE {
  IVE_SAD_MODE_MB_4X4 = 0x0,   /*4x4*/
  IVE_SAD_MODE_MB_8X8 = 0x1,   /*8x8*/
  IVE_SAD_MODE_MB_16X16 = 0x2, /*16x16*/

  IVE_SAD_MODE_BUTT
} IVE_SAD_MODE_E;
/*
 *Sad output ctrl
 */
typedef enum IVE_SAD_OUT_CTRL {
  IVE_SAD_OUT_CTRL_16BIT_BOTH = 0x0, /*Output 16 bit sad and thresh*/
  IVE_SAD_OUT_CTRL_8BIT_BOTH = 0x1,  /*Output 8 bit sad and thresh*/
  IVE_SAD_OUT_CTRL_16BIT_SAD = 0x2,  /*Output 16 bit sad*/
  IVE_SAD_OUT_CTRL_8BIT_SAD = 0x3,   /*Output 8 bit sad*/
  IVE_SAD_OUT_CTRL_THRESH = 0x4,     /*Output thresh,16 bits sad */

  IVE_SAD_OUT_CTRL_BUTT
} IVE_SAD_OUT_CTRL_E;
/*
 * Sad ctrl param
 */
typedef struct IVE_SAD_CTRL {
  IVE_SAD_MODE_E enMode;
  IVE_SAD_OUT_CTRL_E enOutCtrl;
  CVI_U16 u16Thr;  /*srcVal <= u16Thr, dstVal = minVal; srcVal > u16Thr, dstVal = maxVal.*/
  CVI_U8 u8MinVal; /*Min value*/
  CVI_U8 u8MaxVal; /*Max value*/
} IVE_SAD_CTRL_S;

typedef enum IVE_SOBEL_OUT_CTRL {
  IVE_SOBEL_OUT_CTRL_BOTH = 0x0, /*Output horizontal and vertical*/
  IVE_SOBEL_OUT_CTRL_HOR = 0x1,  /*Output horizontal*/
  IVE_SOBEL_OUT_CTRL_VER = 0x2,  /*Output vertical*/
  IVE_SOBEL_OUT_CTRL_BUTT
} IVE_SOBEL_OUT_CTRL_E;

typedef struct IVE_SOBEL_CTRL {
  IVE_SOBEL_OUT_CTRL_E enOutCtrl; /*Output format*/
  CVI_S8 as8Mask[25];             /*Template parameter*/
} IVE_SOBEL_CTRL_S;

typedef enum hiIVE_SUB_MODE_E {
  IVE_SUB_MODE_NORMAL = 0x0,
  IVE_SUB_MODE_ABS = 0x1,   /*Absolute value of the difference*/
  IVE_SUB_MODE_SHIFT = 0x2, /*The output result is obtained by shifting the result one digit right
                               to reserve the signed bit.*/
  IVE_SUB_MODE_BUTT
} IVE_SUB_MODE_E;

typedef struct IVE_SUB_CTRL {
  IVE_SUB_MODE_E enMode;
} IVE_SUB_CTRL_S;

typedef enum IVE_THRESH_MODE { IVE_THRESH_MODE_BINARY } IVE_THRESH_MODE_E;

typedef struct IVE_THRESH_CTRL {
  CVI_U32 enMode;
  CVI_U8 u8MinVal;
  CVI_U8 u8MaxVal;
  CVI_U8 u8LowThr;
} IVE_THRESH_CTRL_S;

#endif  // End of _CVI_COMM_IVE.h