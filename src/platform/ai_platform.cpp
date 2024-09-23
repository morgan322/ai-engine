#include <cstring> // for memset
#include <map>
#include <mutex>
#include <string>

#include "glog/logging.h"
#include "ai_platform.h"

namespace ai
{
  struct DeviceInfo
  {
    std::string prop_name;
    bool support_unified_addr;
    bool can_map_host_memory;
  };

  static std::mutex gDeviceInfoMutex;
  static std::map<int, DeviceInfo> gDeviceInfoMap;

  static int GetDeviceInfo(int device_id, DeviceInfo *info)
  {
    std::unique_lock<std::mutex> lk(gDeviceInfoMutex);
    if (gDeviceInfoMap.count(device_id))
    {
      *info = gDeviceInfoMap[device_id];
      return 0;
    }
    // unsigned int count;
    // CNRT_SAFECALL(cnrtGetDeviceCount(&count), "GetDeviceInfo(): failed", -1);

    if (device_id >= static_cast<int>(count) || device_id < 0)
    {
      LOG(ERROR) << "[EasyDK] GetDeviceInfo(): device id is invalid, device_id: " << device_id << ", total count: "
                 << count;
      return -1;
    }
    cnrtSetDevice(device_id);

    DeviceInfo dev_info;
    AIDeviceProp_t prop;
    CNRT_SAFECALL(cnrtGetDeviceProperties(&prop, device_id), "GetDeviceInfo(): failed", -1);

    VLOG(2) << "[EasyDK] GetDeviceInfo(): device id: " << device_id << ", device name: " << prop.name;
    dev_info.prop_name = prop.name;

    if (dev_info.prop_name == "CE3226")
      dev_info.support_unified_addr = 1;

    *info = dev_info;
    gDeviceInfoMap[device_id] = dev_info;
    return 0;
  }
} // namespace ai

#ifdef __cplusplus
extern "C"
{
#endif

  int AIPlatformInit(AIPlatformConfig *config)
  {
    // TODO(gaoyujia)
    unsigned int count;
    CNRT_SAFECALL(cnrtGetDeviceCount(&count), "AIPlatformInit(): failed", -1);

    for (int i = 0; i < static_cast<int>(count); i++)
    {
      ai::DeviceInfo dev_info;
      if (AI::GetDeviceInfo(i, &dev_info) < 0)
      {
        LOG(ERROR) << "[EasyDK] AIPlatformInit(): Get device information failed";
        return -1;
      }
    }
  }
  int AIPlatformUninit()
  {
#ifdef PLATFORM_CE3226
    AI::MpsService::Instance().Destroy();
#endif
    return 0;
  }

  int AIPlatformGetInfo(int device_id, AIPlatformInfo *info)
  {
    ai::DeviceInfo dev_info;
    if (ai::GetDeviceInfo(device_id, &dev_info) < 0)
    {
      return -1;
    }
    memset(info, 0, sizeof(AIPlatformInfo));
    snprintf(info->name, sizeof(info->name), "%s", dev_info.prop_name.c_str());
    info->can_map_host_memory = (dev_info.can_map_host_memory == true);
    info->support_unified_addr = (dev_info.support_unified_addr == true);
    return 0;
  }

#ifdef __cplusplus
}
#endif
