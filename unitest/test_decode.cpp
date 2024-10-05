/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc. All rights reserved
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
#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "glog/logging.h"
#include <jpeglib.h>

#include "../src/codec/ai_decode.h"
#include "../src/codec/ffmpeg_demuxer.h"
#include "test_base.h"

#include <fstream>
#include <iostream>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <opencv2/opencv.hpp>
extern "C" {
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

static const char *jpeg_file = "../../3rdparty/easydk/unitest/data/1080p.jpg";
static const char *corrupt_jpeg_file =
    "../../3rdparty/easydk/unitest/data/A_6000x3374.jpg";
static const char *h264_file = "../../3rdparty/easydk/unitest/data/img.h264";
static const char *h265_file = "../../3rdparty/easydk/unitest/data/img.hevc";

std::condition_variable cond;
std::mutex mut;
bool decode_done = false;

#ifndef MAX_INPUT_DATA_SIZE
#define MAX_INPUT_DATA_SIZE (25 << 20)
#endif

FILE *p_big_stream = NULL;

static const size_t g_device_id = 0;

static uint8_t *g_data_buffer;
static void *g_surf_pool = nullptr;

void SaveFramesAsJPEG(const uint8_t *h264Data, int dataSize) {
  // 初始化 FFmpeg
  av_register_all();
  avcodec_register_all();

  AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
  AVCodecContext *codecCtx = avcodec_alloc_context3(codec);
  avcodec_open2(codecCtx, codec, nullptr);

  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = const_cast<uint8_t *>(h264Data);
  pkt.size = dataSize;

  AVFrame *frame = av_frame_alloc();
  int frameCount = 0;

  int response = avcodec_send_packet(codecCtx, &pkt);
  while (response >= 0) {
    response = avcodec_receive_frame(codecCtx, frame);
    if (response == 0) {
      // 创建 OpenCV Mat 对象
      cv::Mat img(frame->height, frame->width, CV_8UC3);

      // 转换 YUV 到 BGR
      // 需要使用 Swscale 来转换像素格式
      struct SwsContext *swsCtx =
          sws_getContext(frame->width, frame->height, AV_PIX_FMT_YUV420P,
                         frame->width, frame->height, AV_PIX_FMT_BGR24,
                         SWS_BILINEAR, nullptr, nullptr, nullptr);

      uint8_t *dest[] = {img.data};
      int destLinesize[] = {img.step[0]};
      sws_scale(swsCtx, frame->data, frame->linesize, 0, frame->height, dest,
                destLinesize);

      // 保存每一帧为 JPEG 文件
      std::string filename = "frame_" + std::to_string(frameCount++) + ".jpg";
      cv::imwrite(filename, img);

      sws_freeContext(swsCtx);
    }
  }

  av_frame_free(&frame);
  avcodec_free_context(&codecCtx);
}

bool SendData(void *vdec, AIVdecType type, std::string file,
              bool test_crush = false) {
  if (file == "") {
    LOG(INFO) << "[AI Tests] [Decode] SendData(): Send EOS";
    AIVdecStream stream;
    stream.bits = nullptr;
    stream.len = 0;
    stream.pts = 0;
    // if (AIVdecSendStream(vdec, &stream, 5000) != 0) { // send eos
    //   LOG(ERROR) << "[AI Tests] [Decode] Send Eos Failed";
    //   return false;
    // }
    return true;
  }

  uint64_t pts = 0;
  std::string test_path = GetExePath() + file;
  LOG(INFO) << "[AI Tests] [Decode] SendData(): Send data from " << test_path;
  if (type == AI_VDEC_TYPE_JPEG) {
    FILE *fid;
    fid = fopen(test_path.c_str(), "rb");
    if (fid == NULL) {
      LOG(ERROR) << "[AI Tests] [Decode] SendData(): Open file failed";
      return false;
    }
    fseek(fid, 0, SEEK_END);
    int64_t file_len = ftell(fid);
    rewind(fid);
    if ((file_len == 0) || (file_len > MAX_INPUT_DATA_SIZE)) {
      fclose(fid);
      LOG(ERROR) << "[AI Tests] [Decode] SendData(): File length is 0";
      return false;
    }
    std::vector<unsigned char> buffer(file_len);
    size_t read_length = fread(buffer.data(), 1, file_len, fid);
    fclose(fid);

    AIVdecStream stream;
    memset(&stream, 0, sizeof(stream));

    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    cv::imwrite("decoded_frame.jpg", image);

    stream.bits = g_data_buffer;
    stream.len = (int)read_length;
    stream.pts = pts++;
    // if (AIVdecSendStream(vdec, &stream, 5000) < 0) {
    //   LOG(ERROR) << "[AI Tests] [Decode] SendData(): Send data failed";
    //   return false;
    // }
    return true;
  }

  std::unique_ptr<FFmpegDemuxer> demuxer;
  demuxer =
      std::unique_ptr<FFmpegDemuxer>{new FFmpegDemuxer(test_path.c_str())};
  int corrupt_time = 0;
  while (1) {
    int data_len = 0;
    if (!demuxer->ReadFrame(reinterpret_cast<uint8_t **>(&g_data_buffer),
                            reinterpret_cast<int *>(&data_len))) {
      LOG(INFO) << "[AI Tests] [Decode] SendData(): Read frame failed";
      break;
    }

    LOG(INFO) << "[AI Tests] [Decode] SendData(): Send data len " << data_len;
    if (data_len < 10000)
      continue;
    SaveFramesAsJPEG(g_data_buffer, data_len);

    if (test_crush == true) {
      if (corrupt_time++ == 3)
        test_crush = false;
      continue;
    }

    AIVdecStream stream;
    memset(&stream, 0, sizeof(stream));

    stream.bits = g_data_buffer;
    stream.len = data_len;
    stream.pts = pts++;
    int retry = 5;
    // while ((AIVdecSendStream(vdec, &stream, 5000) < 0) && (retry-- > 0)) {
    //   std::this_thread::sleep_for(std::chrono::microseconds(500));
    // }
    if (retry < 0)
      return false;
  }

  return true;
}

static int CreateSurfacePool(void **surf_pool, int width, int height,
                             AIBufSurfaceColorFormat fmt) {

  AIBufSurfaceCreateParams create_params;
  memset(&create_params, 0, sizeof(create_params));
  create_params.batch_size = 1;
  create_params.width = width;
  create_params.height = height;
  create_params.color_format = fmt;
  create_params.device_id = g_device_id;
  create_params.mem_type = AI_BUF_MEM_DEVICE;

  if (AIBufPoolCreate(surf_pool, &create_params, 6) < 0) {
    LOG(ERROR) << "[AI Tests] [Decode] CreateSurfacePool(): Create pool failed";
    return -1;
  }

  return 0;
}

int GetBufSurface(AIBufSurface **surf, int width, int height,
                  AIBufSurfaceColorFormat fmt, int timeout_ms,
                  void *user_data) {

  // if (AIBufSurfaceCreateFromPool(surf, g_surf_pool) < 0) {
  //   LOG(ERROR) << "[AI Tests] [Decode] GetBufSurface(): Get BufSurface from "
  //                 "pool failed";
  //   return -1;
  // }
  return 0;
}

int OnFrame(AIBufSurface *surf, void *user_data) {
  surf->surface_list[0].width -= surf->surface_list[0].width & 1;
  surf->surface_list[0].height -= surf->surface_list[0].height & 1;
  surf->surface_list[0].plane_params.width[0] -=
      surf->surface_list[0].plane_params.width[0] & 1;
  surf->surface_list[0].plane_params.height[0] -=
      surf->surface_list[0].plane_params.height[0] & 1;
  surf->surface_list[0].plane_params.width[1] -=
      surf->surface_list[0].plane_params.width[1] & 1;
  surf->surface_list[0].plane_params.height[1] -=
      surf->surface_list[0].plane_params.height[1] & 1;

  if (p_big_stream == NULL) {
    p_big_stream = fopen("big.yuv", "wb");
    if (p_big_stream == NULL) {
      return -1;
    }
  }

  size_t length = surf->surface_list[0].data_size;

  uint8_t *buffer = new uint8_t[length];
  if (!buffer)
    return -1;

  // cnrtMemcpy(buffer, surf->surface_list[0].data_ptr,
  //            surf->surface_list[0].width * surf->surface_list[0].height,
  //            cnrtMemcpyDevToHost);
  // cnrtMemcpy(buffer +
  //                surf->surface_list[0].width * surf->surface_list[0].height,
  //            reinterpret_cast<void *>(
  //                reinterpret_cast<uint64_t>(surf->surface_list[0].data_ptr) +
  //                surf->surface_list[0].width * surf->surface_list[0].height),
  //            surf->surface_list[0].width * surf->surface_list[0].height / 2,
  //            cnrtMemcpyDevToHost);

  size_t written;
  written = fwrite(buffer, 1, length, p_big_stream);
  if (written != length) {
    LOG(ERROR) << "[AI Tests] [Decode] Big written size " << written
               << " != data length " << length;
  }

  // AIBufSurfaceDestroy(surf);

  delete[] buffer;
  return 0;
}

int OnEos(void *user_data) {
  LOG(INFO) << "[AI Tests] [Decode] OnEos" << std::endl;
  if (p_big_stream) {
    fflush(p_big_stream);
    fclose(p_big_stream);
    p_big_stream = NULL;
  }
  decode_done = true;
  cond.notify_one();
  return 0;
}

int OnError(int err_code, void *user_data) {
  LOG(INFO) << "[AI Tests] [Decode] OnError";
  return 0;
}

bool Create(void **vdec, AIVdecType type, uint32_t frame_w, uint32_t frame_h,
            AIBufSurfaceColorFormat fmt) {

  AIVdecCreateParams create_params;
  memset(&create_params, 0, sizeof(create_params));
  create_params.device_id = g_device_id;

  create_params.max_width = frame_w;
  create_params.max_height = frame_h;
  create_params.frame_buf_num = 33;
  create_params.surf_timeout_ms = 5000;
  create_params.userdata = nullptr;
  create_params.GetBufSurf = GetBufSurface;
  create_params.OnFrame = OnFrame;
  create_params.OnEos = OnEos;
  create_params.OnError = OnError;
  create_params.type = type;
  create_params.color_format = fmt;

  // int ret = AIVdecCreate(vdec, &create_params);
  // if (ret) {
  //   LOG(ERROR) << "[AI Tests] [Decode] Create decoder failed";
  //   return false;
  // }
  LOG(INFO) << "[AI Tests] [Decode] Create decoder done";
  return true;
}

int TestDecode(std::string file, AIVdecType type, uint32_t frame_w,
               uint32_t frame_h,
               AIBufSurfaceColorFormat fmt = AI_BUF_COLOR_FORMAT_NV21,
               bool test_crush = false) {
  void *decode;
  int ret = 0;
  decode_done = false;
  std::unique_ptr<uint8_t[]> buffer =
      std::unique_ptr<uint8_t[]>{new uint8_t[MAX_INPUT_DATA_SIZE]};
  g_data_buffer = buffer.get();
  if (!g_data_buffer) {
    LOG(ERROR) << "[AI Tests] [Decode] malloc cpu data failed";
    return -1;
  }

  if (!Create(&decode, type, frame_w, frame_h, fmt)) {
    LOG(ERROR) << "[AI Tests] [Decode] Create decode failed";
    return -1;
  }

  if (!SendData(decode, type, file)) {
    LOG(ERROR) << "[AI Tests] [Decode] Send Data failed";
    ret = -1;
  }

  if (!SendData(decode, type, "")) { // send eos
    LOG(ERROR) << "[AI Tests] [Decode] Send Eos failed";
    ret = -1;
  }

  // {
  //   std::unique_lock<std::mutex> lk(mut);
  //   cond.wait(lk, []() -> bool { return decode_done; });
  // }
  // if (AIVdecDestroy(decode) < 0) {
  //   LOG(ERROR) << "[AI Tests] [Decode] Destory decode failed";
  //   ret = -1;
  // }
  return ret;
}

TEST(Decode, Jpeg) {
  EXPECT_EQ(TestDecode(jpeg_file, AI_VDEC_TYPE_JPEG, 1920, 1080), 0);
}

TEST(Decode, H264) {
  EXPECT_EQ(TestDecode(h264_file, AI_VDEC_TYPE_H264, 1920, 1080), 0);
}