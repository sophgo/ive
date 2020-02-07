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
  IVE_SUB_MODE_ABS = 0x0,   /*Absolute value of the difference*/
  IVE_SUB_MODE_SHIFT = 0x1, /*The output result is obtained by shifting the result one digit right
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