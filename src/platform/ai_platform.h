#ifndef AI_PLATFORM_H_
#define AI_PLATFORM_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Holds the parameters of a sensor.
 */
typedef struct AISensorParams {
  /** The sensor type */
  int sensor_type;
  /** The mipi device */
  int mipi_dev;
  /** The bus id */
  int bus_id;
  /** The sns clock id */
  int sns_clk_id;
  /** The width of the output */
  int out_width;
  /** The height of the output */
  int out_height;
  /** Not used (NV12 by default). The color format of the output. */
  int output_format;
} AISensorParams;

/**
 * Holds the parameters of vout.
 */
typedef struct AIVoutParams {
  /** The max width of the input */
  int max_input_width;
  /** The max height of the input */
  int max_input_height;
  /** Not used (NV12 by default). The color format of the input. */
  int input_format;
} AIVoutParams;

/**
 * Holds the configurations of the platform.
 */
typedef struct AIPlatformConfig {
  /** The number of sensors. Only Valid on CE3226 platform */
  int sensor_num;
  /** The parameters of sensors. Only Valid on CE3226 platform */
  AISensorParams *sensor_params;
  /** The parameters of vout. Only Valid on CE3226 platform */
  AIVoutParams *vout_params;
  /** The starting codec id. Only Valid on CE3226 platform */
  int codec_id_start;
} AIPlatformConfig;

#define AI_PLATFORM_NAME_LEN  128
/**
 * Holds the Information of the platform.
 */
typedef struct AIPlatformInfo {
  /** The name of the platform */
  char name[AI_PLATFORM_NAME_LEN];
  /** Whether supporting unified address. 1 means supporting, and 0 means not supporting*/
  int support_unified_addr;
  /** Whether supporting map host memory. 1 means supporting, and 0 means not supporting*/
  int can_map_host_memory;
} AIPlatformInfo;

/**
 * @brief Initializes the platform. On CE3226 platform, vin and vout will be initailzed.
 *
 * @param[in] config The configurations of the platform.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIPlatformInit(AIPlatformConfig *config);
/**
 * @brief UnInitializes the platform. On CE3226 platform, vin and vout will be uninitailzed.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIPlatformUninit();
/**
 * @brief Gets the information of the platform.
 *
 * @param[in] device_id Specified the device id.
 * @param[out] info The information of the platform, including the name of the platfrom and so on.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int AIPlatformGetInfo(int device_id, AIPlatformInfo *info);

#ifdef __cplusplus
}
#endif

#endif  // AI_PLATFORM_H_
