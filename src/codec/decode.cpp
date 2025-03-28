// #include <chrono>
// #include <condition_variable>
// #include <functional>
// #include <memory>
// #include <string>
// #include <thread>
// #include <vector>

// #include "glog/logging.h"

// #include "ai_decode.h"
// #include "decode.hpp"
// // ------------------------------- pipeline --------------------------------------
// namespace ai {
// namespace codec {
// int Decode::Process(std::shared_ptr<ai::module::Frame> frame) {
//   if (!eos_send_) {
//     // cnrtSetDevice(dev_id_);
//     AIVdecStream packet;
//     memset(&packet, 0, sizeof(packet));

//     int data_len = 0;
//     int64_t pts;
//     if (!demuxer_->ReadFrame(reinterpret_cast<uint8_t **>(&packet.bits),
//                              &data_len, &pts)) {
//       // LOG(ERROR) << "[EasyDK Tests] [Decode] ReadFrame(): Read data failed";
//       packet.bits = nullptr;
//       if (AIVdecSendStream(vdec_, &packet, 5000)) { // send eos;
//         LOG(ERROR) << "[AI CODEC] [Decode] SendData(): Send data failed";
//       }
//       eos_send_ = true;
//       return 0;
//     }
//     packet.len = data_len;
//     packet.pts = pts;
//     int retry_time = 50;
//     while (retry_time--) {
//       if (AIVdecSendStream(vdec_, &packet, 5000) < 0) {
//         LOG(ERROR) << "[AI] [Decode] SendData(): Send data failed";
//       } else {
//         break;
//       }
//     }
//   }
//   fr_controller_->Control();
//   return 0;
// }

// int Decode::Close() {
//   if (!vdec_) { // if vdec not created, send eos
//     std::shared_ptr<ai::module::Frame> new_frame =
//         std::make_shared<ai::module::Frame>();
//     new_frame->stream_id = stream_id_;
//     new_frame->is_eos = true;
//     new_frame->frame_idx = frame_count_++;
//     eos_send_ = true;
//     Transmit(new_frame);
//   }

//   if (!eos_send_) {
//     AIVdecStream packet;
//     memset(&packet, 0, sizeof(packet));
//     packet.bits = nullptr;
//     // if (AIVdecSendStream(vdec_, &packet, 5000))
//     // { // send eos;
//     //     LOG(ERROR) << "[AI] [Decode] SendData(): Send data failed";
//     // }
//     eos_send_ = true;
//   }
//   return 0;
// }

// Decode::~Decode() {
//   if (vdec_) {
//     AIVdecDestroy(vdec_);
//     vdec_ = nullptr;
//   }
//   if (surf_pool_) {
//     AIBufPoolDestroy(surf_pool_);
//     surf_pool_ = nullptr;
//   }
// }

// int Decode::Open() {
// //   cnrtSetDevice(dev_id_);
//   demuxer_ = std::unique_ptr<FFmpegDemuxer>{new FFmpegDemuxer()};
//   if (demuxer_ == nullptr) {
//     return -1;
//   }
//   int ret = demuxer_->Open(filename_.c_str());
//   if (ret < 0) {
//     LOG(ERROR) << "[AI] [Decode] Create demuxer failed";
//     return -1;
//   }
//   fr_controller_.reset(new FrController(frame_rate_));

//   params_.color_format = AI_BUF_COLOR_FORMAT_NV21;
//   params_.device_id = dev_id_;
//   switch (demuxer_->GetVideoCodec()) {
//   case AV_CODEC_ID_MJPEG:
//     params_.type = AI_VDEC_TYPE_JPEG;
//     break;
//   case AV_CODEC_ID_H264:
//     params_.type = AI_VDEC_TYPE_H264;
//     break;
//   case AV_CODEC_ID_HEVC:
//     params_.type = AI_VDEC_TYPE_H265;
//     break;
//   default:
//     LOG(ERROR) << "[AI] [Decode] Not support codec type";
//     return -1;
//     break;
//   }

//   params_.userdata = this;
//   params_.frame_buf_num = 12; // for CE3226
//   params_.surf_timeout_ms = 5000;

//   params_.max_width = 3840;
//   params_.max_height = 2160;

//   params_.GetBufSurf = GetBufSurface_;
//   params_.OnEos = OnEos_;
//   params_.OnError = OnError_;
//   params_.OnFrame = OnFrame_;
//   width_ = 1920;
//   height_ = 1080;

//   ret = AIVdecCreate(&vdec_, &params_);
//   if (ret < 0) {
//     LOG(ERROR) << "[AI] [Decode] Create decode failed";
//     return -1;
//   }

//   if (CreateSurfacePool(&surf_pool_, width_, height_) < 0) {
//     LOG(ERROR) << "[AI] [Decode] Create Surface pool failed";
//     return -1;
//   }

//   fr_controller_->Start();

//   return 0;
// }

// int Decode::CreateSurfacePool(void **surf_pool, int width, int height) {
//   AIBufSurfaceCreateParams create_params;
//   memset(&create_params, 0, sizeof(create_params));
//   create_params.batch_size = 1;
//   create_params.width = width;
//   create_params.height = height;
//   create_params.color_format = AI_BUF_COLOR_FORMAT_NV21;
//   create_params.device_id = dev_id_;

//   AIPlatformInfo platform_info;
//   AIPlatformGetInfo(dev_id_, &platform_info);
//   std::string platform(platform_info.name);
//   if (platform == "MLU370") {
//     create_params.mem_type = AI_BUF_MEM_DEVICE;
//   } else if (platform == "CE3226") {
//     create_params.mem_type = AI_BUF_MEM_VB_CACHED;
//   }

//   if (AIBufPoolCreate(surf_pool, &create_params, 12) < 0) {
//     LOG(ERROR) << "[AI] [Decode] CreateSurfacePool(): Create pool failed";
//     return -1;
//   }
//   return 0;
// }

// int Decode::OnError(int err_code) {
//   LOG(ERROR) << "[AI] [Decode] OnError";
//   return 0;
// }

// int Decode::OnFrame(AIBufSurface *surf) {
//   if (surf != nullptr) {
//     surf->surface_list[0].width -= surf->surface_list[0].width & 1;
//     surf->surface_list[0].height -= surf->surface_list[0].height & 1;
//     surf->surface_list[0].plane_params.width[0] -=
//         surf->surface_list[0].plane_params.width[0] & 1;
//     surf->surface_list[0].plane_params.height[0] -=
//         surf->surface_list[0].plane_params.height[0] & 1;
//     surf->surface_list[0].plane_params.width[1] -=
//         surf->surface_list[0].plane_params.width[1] & 1;
//     surf->surface_list[0].plane_params.height[1] -=
//         surf->surface_list[0].plane_params.height[1] & 1;

//     std::shared_ptr<ai::basic::Frame> new_frame =
//         std::make_shared<ai::basic::Frame>();
//     new_frame->stream_id = stream_id_;
//     new_frame->is_eos = false;
//     // new_frame->surf = std::make_shared<ai::BufSurfaceWrapper>(surf);

//     new_frame->frame_idx = frame_count_++;
//     // AIBufSurfaceDestroy(surf);

//     Transmit(new_frame);
//   }

//   // size_t length = surf->surface_list[0].data_size;
//   // LOG(INFO) << "length: " << length;
//   return 0;
// }

// int Decode::OnEos() {
//   LOG(INFO) << "[AI] [Decode] OnEos" << std::endl;
//   std::shared_ptr<ai::basic::Frame> new_frame =
//       std::make_shared<ai::basic::Frame>();
//   new_frame->stream_id = stream_id_;
//   new_frame->is_eos = true;
//   new_frame->frame_idx = frame_count_++;
//   Transmit(new_frame);

//   return 0;
// }

// // int Decode::GetBufSurface(AIBufSurface **surf, int width, int height,
// //                           AIBufSurfaceColorFormat fmt, int timeout_ms)
// // {
// //     if (surf_pool_)
// //     {
// //         int retry_time = 50;
// //         while (retry_time--)
// //         {
// //             if (AIBufSurfaceCreateFromPool(surf, surf_pool_) == 0)
// //             {
// //                 break;
// //             }
// //             std::this_thread::sleep_for(std::chrono::milliseconds(500));
// //         }
// //         if (retry_time < 0)
// //         {
// //             LOG(ERROR) << "[AI] [Decode] GetBufSurface(): Get BufSurface from
// //             pool failed"; return -1;
// //         }
// //     }
// //     return 0;
// // }
// } // namespace codec
// } // namespace ai
