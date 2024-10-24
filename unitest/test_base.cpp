#include "test_base.h"

#include <unistd.h>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>

#define PATH_MAX_LENGTH 1024

std::string GetExePath() {
  char path[PATH_MAX_LENGTH];
  int cnt = readlink("/proc/self/exe", path, PATH_MAX_LENGTH);
  if (cnt < 1 || cnt >= PATH_MAX_LENGTH) {
    return "";
  }
  if (path[cnt - 1] == '/') {
    path[cnt - 1] = '\0';
  } else {
    path[cnt] = '\0';
  }
  std::string result(path);
  return std::string(path).substr(0, result.find_last_of('/') + 1);
}

// std::string GetModelInfoStr(std::string model_name, std::string info_type) {
//   CnedkPlatformInfo platform_info;
//   CnedkPlatformGetInfo(g_device_id, &platform_info);
//   std::string platform_name(platform_info.name);
//   std::string model_key;
//   if (platform_name.rfind("MLU5", 0) == 0) {
//     model_key = model_name + "_MLU590";
//   } else {
//     model_key = model_name + "_" + platform_name;
//   }
//   if (g_model_info.find(model_key) != g_model_info.end()) {
//     if (info_type == "name") {
//       return g_model_info.at(model_key).first;
//     } else {
//       return g_model_info.at(model_key).second;
//     }
//   }
//   return "";
// }
