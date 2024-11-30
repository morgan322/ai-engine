#ifndef INFER_SERVER_UTIL_ENV_H_
#define INFER_SERVER_UTIL_ENV_H_

#ifdef _MSC_VER
#include "windows.h"
#else
#include "sys/sysinfo.h"
#endif

#include <cstdlib>
#include <cstring>
#include <string>

namespace infer_server {

/**
 * @brief Get bool value from environment variable
 *
 * @note "false" and "0" means false, otherwise true
 * @param env_str Name of environment variable
 * @param default_val Default value if specified environment variable is not set
 * @return bool Bool value in environment variable
 */
inline bool GetBoolFromEnv(const std::string& env_str, bool default_val = false) noexcept {
  char* var = std::getenv(env_str.c_str());
  if (!var) return default_val;

  if (strcmp(var, "false") == 0 || strcmp(var, "0") == 0) {
    return false;
  } else {
    return true;
  }
}

/**
 * @brief Get int value from environment variable
 *
 * @exception Same as std::stoi
 * @param env_str Name of environment variable
 * @param default_val Default value if specified environment variable is not set
 * @return int Int value in environment variable
 */
inline int GetIntFromEnv(const std::string& env_str, int default_val = 0) {
  char* var = std::getenv(env_str.c_str());
  if (!var) return default_val;

  return std::stoi(var);
}

/**
 * @brief Get ulong value from environment variable
 *
 * @exception Same as std::stuol
 * @param env_str Name of environment variable
 * @param default_val Default value if specified environment variable is not set
 * @return unsigned long Unsigned long value in environment variable
 */
inline unsigned long GetUlongFromEnv(const std::string& env_str, unsigned long default_val = 0) {  // NOLINT
  char* var = std::getenv(env_str.c_str());
  if (!var) return default_val;

  return std::stoul(var);
}

/**
 * @brief Get string value from environment variable
 *
 * @param env_str Name of environment variable
 * @param default_val Default value if specified environment variable is not set
 * @return std::string String value in environment variable
 */
inline std::string GetStringFromEnv(const std::string& env_str, const std::string& default_val = "") noexcept {
  char* var = std::getenv(env_str.c_str());
  if (!var) return default_val;
  return std::string(var);
}

/**
 * @brief Get CPU core number
 *
 * @return int CPU core number
 */
inline int GetCpuCoreNumber() {
#ifdef _MSC_VER
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  return sysInfo.dwNumberOfProcessors;
#else
  /* return sysconf(_SC_NPROCESSORS_ONLN); */

  // GNU way
  return get_nprocs();
#endif
}

}  // namespace infer_server

#endif  // INFER_SERVER_UTIL_ENV_H_
