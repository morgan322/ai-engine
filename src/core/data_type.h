// #ifndef INFER_SERVER_CORE_DATATYPE_H_
// #define INFER_SERVER_CORE_DATATYPE_H_

// #include <glog/logging.h>
// #include <string>
// #include <vector>

// #include "inference/infer_server.h"
// // #include "cnrt.h"

// // magicmind
// #ifdef HAVE_MM_COMMON_HEADER
// #include "mm_common.h"
// #include "mm_runtime.h"
// #else
// // #include "common.h"
// // #include "interface_runtime.h"
// #endif

// namespace infer_server {
// namespace detail {

// inline std::string DataTypeStr(DataType type) noexcept {
//   switch (type) {
// #define DATATYPE2STR(type) \
//   case DataType::type:     \
//     return #type;
//     DATATYPE2STR(UINT8)
//     DATATYPE2STR(FLOAT16)
//     DATATYPE2STR(FLOAT32)
//     DATATYPE2STR(INT32)
//     DATATYPE2STR(INT16)
// #undef DATATYPE2STR
//     default:
//       LOG(ERROR) << "[EasyDK InferServer] DataTypeStr(): Unsupported data type";
//       return "INVALID";
//   }
// }

// inline std::string DimOrderStr(DimOrder order) noexcept {
//   switch (order) {
// #define DIMORDER2STR(order) \
//   case DimOrder::order:     \
//     return #order;
//     DIMORDER2STR(NCHW)
//     DIMORDER2STR(NHWC)
//     DIMORDER2STR(HWCN)
//     DIMORDER2STR(TNC)
//     DIMORDER2STR(NTC)
//     DIMORDER2STR(NONE)
//     DIMORDER2STR(ARRAY)
// #undef DIMORDER2STR
//     default:
//       LOG(ERROR) << "[EasyDK InferServer] DimOrderStr(): Unsupported dim order";
//       return "INVALID";
//   }
// }

// inline cnrtDataType CastDataType(DataType type) noexcept {
//   switch (type) {
// #define RETURN_DATA_TYPE(type) \
//   case DataType::type:         \
//     return CNRT_##type;
//     RETURN_DATA_TYPE(UINT8)
//     RETURN_DATA_TYPE(FLOAT16)
//     RETURN_DATA_TYPE(FLOAT32)
//     RETURN_DATA_TYPE(INT32)
//     RETURN_DATA_TYPE(INT16)
// #undef RETURN_DATA_TYPE
//     default:
//       LOG(ERROR) << "[EasyDK InferServer] CastDataType(): Unsupported data type";
//       return CNRT_INVALID;
//   }
// }

// inline DataType CastDataType(magicmind::DataType type) noexcept {
//   switch (type) {
// #define RETURN_DATA_TYPE(type)    \
//   case magicmind::DataType::type: \
//     return DataType::type;
//     RETURN_DATA_TYPE(UINT8)
//     RETURN_DATA_TYPE(FLOAT16)
//     RETURN_DATA_TYPE(FLOAT32)
//     RETURN_DATA_TYPE(INT32)
//     RETURN_DATA_TYPE(INT16)
// #undef RETURN_DATA_TYPE
//     default:
//       LOG(ERROR) << "[EasyDK InferServer] CastDataType(): Unsupported MagicMind data type";
//       return DataType::INVALID;
//   }
// }

// inline DataType CastDataType(cnrtDataType type) noexcept {
//   switch (type) {
// #define RETURN_DATA_TYPE(type) \
//   case CNRT_##type:            \
//     return DataType::type;
//     RETURN_DATA_TYPE(UINT8)
//     RETURN_DATA_TYPE(FLOAT16)
//     RETURN_DATA_TYPE(FLOAT32)
//     RETURN_DATA_TYPE(INT32)
//     RETURN_DATA_TYPE(INT16)
// #undef RETURN_DATA_TYPE
//     default:
//       LOG(ERROR) << "[EasyDK InferServer] CastDataType(): Unsupported CNRT data type";
//       return DataType::INVALID;
//   }
// }

// inline DimOrder CastDimOrder(magicmind::Layout order) noexcept {
//   switch (order) {
// #define RETURN_DIM_ORDER(order)    \
//   case magicmind::Layout::order: \
//     return DimOrder::order;
//     RETURN_DIM_ORDER(NCHW)
//     RETURN_DIM_ORDER(NHWC)
//     RETURN_DIM_ORDER(HWCN)
//     RETURN_DIM_ORDER(TNC)
//     RETURN_DIM_ORDER(NTC)
// #if MM_MAJOR_VERSION <= 0 && MM_MINOR_VERSION < 14
//     RETURN_DIM_ORDER(NONE)
// #else
//     RETURN_DIM_ORDER(ARRAY)
// #endif
// #undef RETURN_DIM_ORDER
//     default:
//       LOG(ERROR) << "[EasyDK InferServer] CastDimOrder(): Unsupported dim order: " << static_cast<int>(order);
//       return DimOrder::INVALID;
//   }
// }

// // shape corresponding to src_data
// bool CastDataType(void *src_data, void *dst_data, DataType src_dtype, DataType dst_dtype, const Shape &shape);

// }  // namespace detail

// template <typename dtype>
// inline std::vector<dtype> DimNHWC2NCHW(const std::vector<dtype>& dim) {
//   switch (dim.size()) {
//     case 1:
//       return dim;
//     case 2:
//       return dim;
//     case 3:
//       return std::vector<dtype>({dim[0], dim[2], dim[1]});
//     case 4:
//       return std::vector<dtype>({dim[0], dim[3], dim[1], dim[2]});
//     case 5:
//       return std::vector<dtype>({dim[0], dim[4], dim[1], dim[2], dim[3]});
//     default:
//       CHECK(0) << "[EasyDK InferServer] DimNHWC2NCHW(): Unsupported dimension";
//   }
//   return {};
// }

// template <typename dtype>
// inline std::vector<dtype> DimNCHW2NHWC(const std::vector<dtype>& dim) {
//   switch (dim.size()) {
//     case 1:
//       return dim;
//     case 2:
//       return dim;
//     case 3:
//       return std::vector<dtype>({dim[0], dim[2], dim[1]});
//     case 4:
//       return std::vector<dtype>({dim[0], dim[2], dim[3], dim[1]});
//     case 5:
//       return std::vector<dtype>({dim[0], dim[2], dim[3], dim[4], dim[1]});
//     default:
//       CHECK(0) << "[EasyDK InferServer] DimNCHW2NHWC(): Unsupported dimension";
//   }
//   return {};
// }

// template <typename T>
// std::ostream& operator<<(std::ostream& os, const std::vector<T>& v_t) {
//   os << "vec {";
//   if (v_t.size()) {
//     for (size_t i = 0; i < v_t.size() - 1; ++i) {
//       os << v_t[i] << ", ";
//     }
//     if (v_t.size() > 0) os << v_t[v_t.size() - 1];
//   }
//   os << "}";
//   return os;
// }

// inline std::string StatusStr(Status status) noexcept {
//   switch (status) {
// #define STATUS2STR(status) \
//   case Status::status:     \
//     return #status;
//     STATUS2STR(SUCCESS)
//     STATUS2STR(ERROR_READWRITE)
//     STATUS2STR(ERROR_MEMORY)
//     STATUS2STR(INVALID_PARAM)
//     STATUS2STR(WRONG_TYPE)
//     STATUS2STR(ERROR_BACKEND)
//     STATUS2STR(NOT_IMPLEMENTED)
//     STATUS2STR(TIMEOUT)
// #undef STATUS2STR
//     default:
//       LOG(ERROR) << "[EasyDK InferServer] [StatusStr] Unsupported Status";
//       return "INVALID";
//   }
// }

// }  // namespace infer_server

// #endif  // INFER_SERVER_CORE_DATATYPE_H_
