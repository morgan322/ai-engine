#ifndef EDK_CONFIG_H_
#define EDK_CONFIG_H_

#include <string>

#define EDK_VERSION_MAJOR 0
#define EDK_VERSION_MINOR 0
#define EDK_VERSION_PATCH 1

#define EDK_GET_VERSION(major, minor, patch) (((major) << 20) | ((minor) << 10) | (patch))
#define EDK_VERSION EDK_GET_VERSION(EDK_VERSION_MAJOR, EDK_VERSION_MINOR, EDK_VERSION_PATCH)

namespace ai {

/**
 * @brief Get edk version string
 *
 * @return std::string version string
 */
inline static std::string Version() {
  // clang-format off
  return std::to_string(EDK_VERSION_MAJOR) + "." +
         std::to_string(EDK_VERSION_MINOR) + "." +
         std::to_string(EDK_VERSION_PATCH);
  // clang-format on
}

}  // namespace ai

#endif  // EDK_CONFIG_H_
