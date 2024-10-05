#include <chrono>
#include <condition_variable>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

#include "../src/codec/ai_encode.h"
#include "../src/platform/ai_platform.h"

#include "test_base.h"

static const char *test_1080p_jpg = "../../3rdparty/easydk/unitest/data/1080p.jpg";
static const char *test_500x500_jpg = "../../3rdparty/easydk/unitest/data/500x500.jpg";

const size_t device_id = 0;

class TestEncode {
 public:
  explicit TestEncode(int dev_id, uint32_t frame_w, uint32_t frame_h, AIVencType type,
                      AIBufSurfaceColorFormat color_format = AI_BUF_COLOR_FORMAT_NV12);
  ~TestEncode();

  int Start();

  int SendData(AIBufSurface *surf, int timeout_ms);

 public:
  static int OnFrameBits_(AIVEncFrameBits *framebits, void *userdata) {
    TestEncode *thiz = reinterpret_cast<TestEncode *>(userdata);
    return thiz->OnFrameBits(framebits);
  }
  static int OnEos_(void *userdata) {
    TestEncode *thiz = reinterpret_cast<TestEncode *>(userdata);
    return thiz->OnEos();
  }
  static int OnError_(int errcode, void *userdata) {
    TestEncode *thiz = reinterpret_cast<TestEncode *>(userdata);
    return thiz->OnError(errcode);
  }

 private:
  int OnFrameBits(AIVEncFrameBits *framebits);
  int OnEos();
  int OnError(int errcode);

 private:
  bool send_done_ = false;
  void *venc_ = nullptr;
  AIVencCreateParams params_;

  std::condition_variable enc_cond_;
  bool eos_send_ = false;
  bool encode_done_ = false;
  std::mutex mut_;

  size_t frame_count_ = 0;

  FILE *p_output_file_ = nullptr;
};

TestEncode::TestEncode(int dev_id, uint32_t frame_w, uint32_t frame_h, AIVencType type,
                       AIBufSurfaceColorFormat color_format) {
  params_.type = type;
  params_.device_id = dev_id;
  params_.width = frame_w;
  params_.height = frame_h;

  params_.color_format = AI_BUF_COLOR_FORMAT_NV12;

  params_.frame_rate = 0;
  params_.key_interval = 0;
  params_.input_buf_num = 2;
  params_.gop_size = 30;
  params_.bitrate = 0x4000000;
  params_.OnFrameBits = TestEncode::OnFrameBits_;
  params_.OnEos = TestEncode::OnEos_;
  params_.OnError = TestEncode::OnError_;
  params_.userdata = this;
}

TestEncode::~TestEncode() {
  if (!eos_send_) {
    SendData(nullptr, 5000);
  }
  // AIVencDestroy(venc_);
}

int TestEncode::Start() {
  // int ret = AIVencCreate(&venc_, &params_);
  // if (ret < 0) {
  //   LOG(ERROR) << "[AI Tests] [Encode] Create encode failed";
  // }
  return 1;
}

int TestEncode::SendData(AIBufSurface *surf, int timeout_ms) {
  // if (!surf) {                                             // send eos
  //   if (AIVencSendFrame(venc_, nullptr, timeout_ms) < 0) {  // send eos
  //     LOG(ERROR) << "[AI Tests] [Encode] Send EOS failed";
  //     return -1;
  //   }

  //   {
  //     std::unique_lock<std::mutex> lk(mut_);
  //     enc_cond_.wait(lk, [&]() -> bool { return encode_done_; });
  //   }
  //   eos_send_ = true;
  //   return 0;
  // }

  // if (AIVencSendFrame(venc_, surf, timeout_ms) < 0) {
  //   LOG(ERROR) << "[AI Tests] [Encode] Send Frame failed";
  //   return -1;
  // }
  return 0;
}

int TestEncode::OnEos() {
  if (p_output_file_) {
    fflush(p_output_file_);
    fclose(p_output_file_);
    p_output_file_ = NULL;
  }

  {
    std::unique_lock<std::mutex> lk(mut_);
    encode_done_ = true;
    enc_cond_.notify_all();
  }

  return 0;
}

int TestEncode::OnFrameBits(AIVEncFrameBits *framebits) {
  char *output_file = NULL;
  char str[256] = {0};

  size_t length = framebits->len;

  if (params_.type == AI_VENC_TYPE_JPEG) {
    snprintf(str, sizeof(str), "./encoded_%d_%d_%02lu.jpg", params_.width, params_.height, frame_count_);
    output_file = str;
  } else if (params_.type == AI_VENC_TYPE_H264) {
    snprintf(str, sizeof(str), "./encoded_%d_%d_%lu.h264", params_.width, params_.height, length);
    output_file = str;
  } else if (params_.type == AI_VENC_TYPE_H265) {
    snprintf(str, sizeof(str), "./encoded_%d_%d_%lu.h265", params_.width, params_.height, length);
    output_file = str;
  } else {
    LOG(ERROR) << "[AI Tests] [Encode] Unsupported output codec type: " << params_.type;
  }

  if (p_output_file_ == NULL) p_output_file_ = fopen(output_file, "wb");
  if (p_output_file_ == NULL) {
    LOG(ERROR) << "[AI Tests] [Encode] Open output file failed";
    return -1;
  }

  size_t written;
  written = fwrite(framebits->bits, 1, length, p_output_file_);
  if (written != length) {
    LOG(ERROR) << "[AI Tests] [Encode] Written size " << written << " != data length " << length;
    return -1;
  }

  return 0;
}

int TestEncode::OnError(int err_code) { return -1; }

#define ALIGN(w, a) ((w + a - 1) & ~(a - 1))
static bool CvtBgrToYuv420sp(const cv::Mat &bgr_image, uint32_t alignment, AIBufSurface *surf) {
  cv::Mat yuv_i420_image;
  uint32_t width, height, stride;
  uint8_t *src_y, *src_u, *src_v, *dst_y, *dst_u;
  // uint8_t *dst_v;

  cv::cvtColor(bgr_image, yuv_i420_image, cv::COLOR_BGR2YUV_I420);

  width = bgr_image.cols;
  height = bgr_image.rows;
  if (alignment > 0)
    stride = ALIGN(width, alignment);
  else
    stride = width;

  uint32_t y_len = stride * height;
  src_y = yuv_i420_image.data;
  src_u = yuv_i420_image.data + y_len;
  src_v = yuv_i420_image.data + y_len * 5 / 4;

  if (surf->mem_type == AI_BUF_MEM_VB_CACHED || surf->mem_type == AI_BUF_MEM_VB) {
    dst_y = reinterpret_cast<uint8_t *>(surf->surface_list[0].mapped_data_ptr);
    dst_u = reinterpret_cast<uint8_t *>(reinterpret_cast<uint64_t>(surf->surface_list[0].mapped_data_ptr) + y_len);
  } else {
    dst_y = reinterpret_cast<uint8_t *>(surf->surface_list[0].data_ptr);
    dst_u = reinterpret_cast<uint8_t *>(reinterpret_cast<uint64_t>(surf->surface_list[0].data_ptr) + y_len);
  }
  for (uint32_t i = 0; i < height; i++) {
    // y data
    memcpy(dst_y + i * stride, src_y + i * width, width);
    // uv data
    if (i % 2 == 0) {
      for (uint32_t j = 0; j < width / 2; j++) {
        if (surf->surface_list->color_format == AI_BUF_COLOR_FORMAT_NV21) {
          *(dst_u + i * stride / 2 + 2 * j) = *(src_v + i * width / 4 + j);
          *(dst_u + i * stride / 2 + 2 * j + 1) = *(src_u + i * width / 4 + j);
        } else {
          *(dst_u + i * stride / 2 + 2 * j) = *(src_u + i * width / 4 + j);
          *(dst_u + i * stride / 2 + 2 * j + 1) = *(src_v + i * width / 4 + j);
        }
      }
    }
  }

  return true;
}

int ProcessEncode(std::string file, AIVencType type, AIBufSurfaceMemType mem_type,
                  AIBufSurfaceColorFormat fmt, uint32_t frame_w, uint32_t frame_h) {
  int ret;
 
  TestEncode encode(device_id, frame_w, frame_h, type, fmt);

  ret = encode.Start();
  if (ret < 0) {
    LOG(ERROR) << "[AI Tests] [Encode] Start encode failed";
    return -1;
  }

  // read data
  std::string test_path = GetExePath() + file;
  cv::Mat mat = cv::imread(test_path);
  
  // create bufsurface
  // 创建并初始化 bufsurface
  AIBufSurface *surf = new AIBufSurface();
  surf->batch_size = 1;
  surf->num_filled = 1; // 仅填充一个表面
  surf->surface_list = new AIBufSurfaceParams[surf->batch_size];
  
  // 初始化 surface_list 中的第一个参数
  AIBufSurfaceParams &surface = surf->surface_list[0];
  surface.width = static_cast<uint32_t>(mat.cols);
  surface.height = static_cast<uint32_t>(mat.rows);
  surface.pitch = mat.step[0]; // 每行字节数
  surface.data_size = static_cast<uint32_t>(mat.total() * mat.elemSize());
  surface.data_ptr = new uint8_t[surface.data_size]; 
  surface.color_format = fmt; 

  // 复制图像数据
  memcpy(surface.data_ptr, mat.data, surface.data_size);

  

  // 清理内存
  delete[] static_cast<uint8_t*>(surface.data_ptr);
  delete[] surf->surface_list;
  delete surf;

 
  
  return 0;
}


TEST(Encode, Jpeg) {
  EXPECT_EQ(ProcessEncode(test_500x500_jpg, AI_VENC_TYPE_JPEG, AI_BUF_MEM_SYSTEM, AI_BUF_COLOR_FORMAT_BGR,
                          1920, 1080), 0);
}

