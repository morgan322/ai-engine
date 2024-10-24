#include "ai_decode.h"

#include <cstring>  // for memset
#include <memory>  // for unique_ptr
#include <mutex>   // for call_once

#include "glog/logging.h"
#include "../platform/ai_platform.h"
#include "../node/ai_decode_impl.hpp"

namespace ai {

IDecoder *CreateDecoder() {
  // cpu sofe decode
  // int dev_id = -1;
  // CNRT_SAFECALL(cnrtGetDevice(&dev_id), "CreateDecoder(): failed", nullptr);

  // AIPlatformInfo info;
  // if (AIPlatformGetInfo(dev_id, &info) < 0) {
  //   LOG(ERROR) << "[AI] CreateDecoder(): Get platform information failed";
  //   return nullptr;
  // }
}

class DecodeService {
 public:
  static DecodeService &Instance() {
    static std::once_flag s_flag;
    std::call_once(s_flag, [&] { instance_.reset(new DecodeService); });
    return *instance_;
  }

  int Create(void **vdec, AIVdecCreateParams *params) {
    if (!vdec || !params) {
      LOG(ERROR) << "[AI] [DecodeService] Create(): decoder or params pointer is invalid";
      return -1;
    }
    if (CheckParams(params) < 0) {
      LOG(ERROR) << "[AI] [DecodeService] Create(): Parameters are invalid";
      return -1;
    }
    IDecoder *decoder_ = CreateDecoder();
    if (!decoder_) {
      LOG(ERROR) << "[AI] [DecodeService] Create(): new decoder failed";
      return -1;
    }
    if (decoder_->Create(params) < 0) {
      LOG(ERROR) << "[AI] [DecodeService] Create(): Create decoder failed";
      delete decoder_;
      return -1;
    }
    *vdec = decoder_;
    return 0;
  }

  int Destroy(void *vdec) {
    if (!vdec) {
      LOG(ERROR) << "[AI] [DecodeService] Destroy(): Decoder pointer is invalid";
      return -1;
    }
    IDecoder *decoder_ = static_cast<IDecoder *>(vdec);
    decoder_->Destroy();
    delete decoder_;
    return 0;
  }

  int SendStream(void *vdec, const AIVdecStream *stream, int timeout_ms) {
    if (!vdec || !stream) {
      LOG(ERROR) << "[AI] [DecodeService] SendStream(): Decoder or stream pointer is invalid";
      return -1;
    }
    IDecoder *decoder_ = static_cast<IDecoder *>(vdec);
    return decoder_->SendStream(stream, timeout_ms);
  }

 private:
  int CheckParams(AIVdecCreateParams *params) {
    if (params->type <= AI_VDEC_TYPE_INVALID || params->type >= AI_VDEC_TYPE_NUM) {
      LOG(ERROR) << "[AI] [DecodeService] CheckParams(): Unsupported codec type: " << params->type;
      return -1;
    }

    if (params->color_format != AI_BUF_COLOR_FORMAT_NV12 && params->color_format != AI_BUF_COLOR_FORMAT_NV21) {
      LOG(ERROR) << "[AI] [DecodeService] CheckParams(): Unsupported color format: " << params->color_format;
      return -1;
    }

    if (params->OnEos == nullptr || params->OnFrame == nullptr || params->OnError == nullptr ||
        params->GetBufSurf == nullptr) {
      LOG(ERROR) << "[AI] [DecodeService] CheckParams(): OnEos, OnFrame, OnError or GetBufSurf function pointer"
                 << " is invalid";
      return -1;
    }

    // int dev_id = -1;
    // CNRT_SAFECALL(cnrtGetDevice(&dev_id), "[DecodeService] CheckParams(): failed", -1);
    // if (params->device_id != dev_id) {
    //   LOG(ERROR) << "[AI] [DecodeService] CheckParams(): device id of current thread and device id in parameters"
    //              << " are different";
    //   return -1;
    // }

    // TODO(gaoyujia)
    return 0;
  }

 private:
  DecodeService(const DecodeService &) = delete;
  DecodeService(DecodeService &&) = delete;
  DecodeService &operator=(const DecodeService &) = delete;
  DecodeService &operator=(DecodeService &&) = delete;
  DecodeService() = default;

 private:
  static std::unique_ptr<DecodeService> instance_;
};

std::unique_ptr<DecodeService> DecodeService::instance_;

}  // namespace ai

extern "C" {

int AIVdecCreate(void **vdec, AIVdecCreateParams *params) {
  return ai::DecodeService::Instance().Create(vdec, params);
}
int AIVdecDestroy(void *vdec) { return ai::DecodeService::Instance().Destroy(vdec); }
int AIVdecSendStream(void *vdec, const AIVdecStream *stream, int timeout_ms) {
  return ai::DecodeService::Instance().SendStream(vdec, stream, timeout_ms);
}
};