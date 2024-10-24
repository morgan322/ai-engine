#ifndef AI_ENCODE_IMPL_HPP_
#define AI_ENCODE_IMPL_HPP_

#include "codec/ai_encode.h"

namespace ai {
namespace node {

class IEncoder {
public:
  virtual ~IEncoder() {}
  virtual int Create(AIVencCreateParams *params) = 0;
  virtual int Destroy() = 0;
  virtual int SendFrame(AIBufSurface *surf, int timeout_ms) = 0;
};

IEncoder *CreateEncoder();

} // namespace node
} // namespace ai

#endif // AI_ENCODE_IMPL_HPP_
