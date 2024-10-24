#include <glog/logging.h>

#include "inference/infer_server.h"


#define CNRT_SAFECALL(func, val)                                                       \
  do {                                                                                 \
    auto ret = (func);                                                                 \
    if (ret != cnrtSuccess) {                                                          \
      LOG(ERROR) << "[EasyDK InferServer] Call " #func " failed, ret = " << ret; \
      return val;                                                                      \
    }                                                                                  \
  } while (0)

namespace infer_server {

bool SetCurrentDevice(int device_id) noexcept {
  CNRT_SAFECALL(cnrtSetDevice(device_id), false);
  VLOG(3) << "[EasyDK InferServer] SetCurrentDevice(): Set device [" << device_id << "] for this thread";
  return true;
}

uint32_t TotalDeviceCount() noexcept {
  uint32_t dev_cnt;
  CNRT_SAFECALL(cnrtGetDeviceCount(&dev_cnt), 0);
  return dev_cnt;
}

bool CheckDevice(int device_id) noexcept {
  uint32_t dev_cnt;
  CNRT_SAFECALL(cnrtGetDeviceCount(&dev_cnt), false);
  return device_id < static_cast<int>(dev_cnt) && device_id >= 0;
}

}  // namespace infer_server
