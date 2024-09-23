/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

/**
 * @file AI_buf_surface.h
 * <b>AIBufSurface Interface </b>
 *
 * This file specifies the AIBufSurface management API.
 *
 * The AIBufSurface API provides methods to allocate / deallocate and copy batched buffers.
 */

#ifndef AI_BUF_SURFACE_H_
#define AI_BUF_SURFACE_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Defines the default padding length for reserved fields of structures. */
#define AI_PADDING_LENGTH  4

/** Defines the maximum number of planes. */
#define AI_BUF_MAX_PLANES   3

/**
 * Specifies color formats for \ref AIBufSurface.
 */
typedef enum {
  /** Specifies an invalid color format. */
  AI_BUF_COLOR_FORMAT_INVALID,
  /** Specifies 8 bit GRAY scale - single plane */
  AI_BUF_COLOR_FORMAT_GRAY8,
  /** Specifies BT.601 colorspace - YUV420 multi-planar. */
  AI_BUF_COLOR_FORMAT_YUV420,
  /** Specifies BT.601 colorspace - Y/CbCr 4:2:0 multi-planar. */
  AI_BUF_COLOR_FORMAT_NV12,
  /** Specifies BT.601 colorspace - Y/CbCr 4:2:0 multi-planar. */
  AI_BUF_COLOR_FORMAT_NV21,
  /** Specifies ARGB-8-8-8-8 single plane. */
  AI_BUF_COLOR_FORMAT_ARGB,
  /** Specifies ABGR-8-8-8-8 single plane. */
  AI_BUF_COLOR_FORMAT_ABGR,
  /** Specifies RGB-8-8-8 single plane. */
  AI_BUF_COLOR_FORMAT_RGB,
  /** Specifies BGR-8-8-8 single plane. */
  AI_BUF_COLOR_FORMAT_BGR,
  /** Specifies BGRA-8-8-8-8 single plane. */
  AI_BUF_COLOR_FORMAT_BGRA,
  /** Specifies RGBA-8-8-8-8 single plane. */
  AI_BUF_COLOR_FORMAT_RGBA,
  /*TODO(gaoyujia): add more color format
  */
  /** Specifies ARGB-1-5-5-5 single plane. */
  AI_BUF_COLOR_FORMAT_ARGB1555,

  /** for inference*/
  AI_BUF_COLOR_FORMAT_TENSOR,
  AI_BUF_COLOR_FORMAT_LAST,
} AIBufSurfaceColorFormat;

/**
 * Specifies memory types for \ref AIBufSurface.
 */
typedef enum {
  /** Specifies the default memory type, i.e. \ref AI_BUF_MEM_DEVICE
   for MLUxxx, \ref AI_BUF_MEM_UNIFIED for CExxxx. Use \ref AI_BUF_MEM_DEFAULT
   to allocate whichever type of memory is appropriate for the platform. */
  AI_BUF_MEM_DEFAULT,
  /** Specifies MLU Device memory type. valid only for MLUxxx*/
  AI_BUF_MEM_DEVICE,
  /** Specifies Host memory type. valid only for MLUxxx */
  AI_BUF_MEM_PINNED,
  /** Specifies Unified memory type. Valid only for CExxxx. */
  AI_BUF_MEM_UNIFIED,
  AI_BUF_MEM_UNIFIED_CACHED,
  /** Specifies VB memory type. Valid only for CExxxx. */
  AI_BUF_MEM_VB,
  AI_BUF_MEM_VB_CACHED,
  /** Specifies memory allocated by malloc(). */
  AI_BUF_MEM_SYSTEM,
} AIBufSurfaceMemType;

/**
 * Holds the planewise parameters of a buffer.
 */
typedef struct AIBufSurfacePlaneParams {
  /** Holds the number of planes. */
  uint32_t num_planes;
  /** Holds the widths of planes. */
  uint32_t width[AI_BUF_MAX_PLANES];
  /** Holds the heights of planes. */
  uint32_t height[AI_BUF_MAX_PLANES];
  /** Holds the pitches of planes in bytes. */
  uint32_t pitch[AI_BUF_MAX_PLANES];
  /** Holds the offsets of planes in bytes. */
  uint32_t offset[AI_BUF_MAX_PLANES];
  /** Holds the sizes of planes in bytes. */
  uint32_t psize[AI_BUF_MAX_PLANES];
  /** Holds the number of bytes occupied by a pixel in each plane. */
  uint32_t bytes_per_pix[AI_BUF_MAX_PLANES];

  void * _reserved[AI_PADDING_LENGTH * AI_BUF_MAX_PLANES];
} AIBufSurfacePlaneParams;

/**
 * Holds parameters required to allocate an \ref AIBufSurface.
 */
typedef struct AIBufSurfaceCreateParams {
  /** Holds the type of memory to be allocated. Not valid for AI_BUF_MEM_VB* */
  AIBufSurfaceMemType mem_type;

  /** Holds the Device ID. */
  uint32_t device_id;
  /** Holds the width of the buffer. */
  uint32_t width;
  /** Holds the height of the buffer. */
  uint32_t height;
  /** Holds the color format of the buffer. */
  AIBufSurfaceColorFormat color_format;

  /** Holds the amount of memory to be allocated. Optional; if set, all other
   parameters (width, height, etc.) are ignored. */
  uint32_t size;

  /** Holds the batch size. */
  uint32_t batch_size;

  /** Holds the alignment mode, if set,  1 bytes alignment will be applied;
   Not valid for AI_BUF_MEM_VB and AI_BUF_MEM_VB_CACHED.
  */
  bool force_align_1;

  void *_reserved[AI_PADDING_LENGTH];
} AIBufSurfaceCreateParams;

/**
 * Holds information about a single buffer in a batch.
 */
typedef struct AIBufSurfaceParams {
  /** Holds the width of the buffer. */
  uint32_t width;
  /** Holds the height of the buffer. */
  uint32_t height;
  /** Holds the pitch of the buffer. */
  uint32_t pitch;
  /** Holds the color format of the buffer. */
  AIBufSurfaceColorFormat color_format;

  /** Holds the amount of allocated memory. */
  uint32_t data_size;

  /** Holds a pointer to allocated memory. */
  void * data_ptr;

  /** Holds a pointer to a CPU mapped buffer.
  Valid only for AI_BUF_MEM_UNIFIED* and AI_BUF_MEM_VB* */
  void * mapped_data_ptr;

  /** Holds planewise information (width, height, pitch, offset, etc.). */
  AIBufSurfacePlaneParams plane_params;

  void * _reserved[AI_PADDING_LENGTH];
} AIBufSurfaceParams;

/**
 * Holds information about batched buffers.
 */
typedef struct AIBufSurface {
  /** Holds type of memory for buffers in the batch. */
  AIBufSurfaceMemType mem_type;

  /** Holds a Device ID. */
  uint32_t device_id;
  /** Holds the batch size. */
  uint32_t batch_size;
  /** Holds the number valid and filled buffers. Initialized to zero when
   an instance of the structure is created. */
  uint32_t num_filled;

  /** Holds an "is contiguous" flag. If set, memory allocated for the batch
  is contiguous. Not valid for AI_BUF_MEM_VB on CE3226 */
  bool is_contiguous;

  /** Holds a pointer to an array of batched buffers. */
  AIBufSurfaceParams *surface_list;

  /** Holds a pointer to the buffer pool context */
  void *opaque;

  /** Holds the timestamp for video image, valid only for batch_size == 1 */
  uint64_t pts;

  void * _reserved[AI_PADDING_LENGTH];
} AIBufSurface;

/**
 * @brief  Creates a Buffer Pool.
 *
 * Call AIBufPoolDestroy() to free resources allocated by this function.
 *
 * @param[out] pool         An indirect pointer to the buffer pool.
 * @param[in]  params       A pointer to an \ref AIBufSurfaceCreateParams
 *                           structure.
 * @param[in]  block_num    The block number.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIBufPoolCreate(void **pool, AIBufSurfaceCreateParams *params, uint32_t block_num);

/**
 * @brief  Frees the buffer pool previously allocated by AIBufPoolCreate().
 *
 * @param[in] surf  A pointer to an \ref buffer pool to be freed.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIBufPoolDestroy(void *pool);

/**
 * @brief  Allocates a single buffer.
 *
 * Allocates memory for a buffer and returns a pointer to an
 * allocated \ref AIBufSurface. The \a params structure must have
 * the allocation parameters of a single buffer.
 *
 * Call AIBufSurfaceDestroy() to free resources allocated by this function.
 *
 * @param[out] surf         An indirect pointer to the allocated buffer.
 * @param[in]  pool         A pointer to a buffer pool.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIBufSurfaceCreateFromPool(AIBufSurface **surf, void *pool);

/**
 * @brief  Allocates a batch of buffers.
 *
 * Allocates memory for \a batch_size buffers and returns a pointer to an
 * allocated \ref AIBufSurface. The \a params structure must have
 * the allocation parameters. If \a params.size
 * is set, a buffer of that size is allocated, and all other
 * parameters (width, height, color format, etc.) are ignored.
 *
 * Call AIBufSurfaceDestroy() to free resources allocated by this function.
 *
 * @param[out] surf         An indirect pointer to the allocated batched
 *                           buffers.
 * @param[in]  params       A pointer to an \ref AIBufSurfaceCreateParams
 *                           structure.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIBufSurfaceCreate(AIBufSurface **surf, AIBufSurfaceCreateParams *params);

/**
 * @brief  Frees a single buffer allocated by AIBufSurfaceCreate()
 *         or batched buffers previously allocated by AIBufSurfaceCreate().
 *
 * @param[in] surf  A pointer to an \ref BufSurface to be freed.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIBufSurfaceDestroy(AIBufSurface *surf);

/**
 * @brief  Syncs the hardware memory cache for the CPU.
 *
 * Valid only for memory types \ref AI_BUF_MEM_UNIFIED_CACHED and
 * \ref AI_BUF_MEM_VB_CACHED.
 *
 * @param[in] surf      A pointer to an \ref NvBufSurface structure.
 * @param[in] index     Index of the buffer in the batch. -1 refers to
 *                      all buffers in the batch.
 * @param[in] plane     Index of a plane in the buffer. -1 refers to all planes
 *                      in the buffer.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIBufSurfaceSyncForCpu(AIBufSurface *surf, int index, int plane);

/**
 * @brief  Syncs the hardware memory cache for the device.
 *
 * Valid only for memory types \ref AI_BUF_MEM_UNIFIED and
 * \ref AI_BUF_MEM_VB.
 *
 * @param[in] surf      A pointer to an \ref AIBufSurface structure.
 * @param[in] index     Index of a buffer in the batch. -1 refers to all buffers
 *                      in the batch.
 * @param[in] plane     Index of a plane in the buffer. -1 refers to all planes
 *                      in the buffer.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIBufSurfaceSyncForDevice(AIBufSurface *surf, int index, int plane);

/**
 * @brief  Copies the content of source batched buffer(s) to destination
 * batched buffer(s).
 *
 * The source and destination \ref AIBufSurface objects must have same
 * buffer and batch size.
 *
 * @param[in] src_surf   A pointer to the source AIBufSurface structure.
 * @param[out] dst_surf   A pointer to the destination AIBufSurface structure.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIBufSurfaceCopy(AIBufSurface *src_surf, AIBufSurface *dst_surf);

/**
 * @brief  Fills each byte of the buffer(s) in an \ref AIBufSurface with a
 * provided value.
 *
 * You can also use this function to reset the buffer(s) in the batch.
 *
 * @param[in] surf  A pointer to the AIBufSurface structure.
 * @param[in] index Index of a buffer in the batch. -1 refers to all buffers
 *                  in the batch.
 * @param[in] plane Index of a plane in the buffer. -1 refers to all planes
 *                  in the buffer.
 * @param[in] value The value to be used as fill.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIBufSurfaceMemSet(AIBufSurface *surf, int index, int plane, uint8_t value);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  // AI_BUF_SURFACE_H_
