
#ifndef CNEDK_DECODE_IMPL_HPP_
#define CNEDK_DECODE_IMPL_HPP_

#include "../codec/ai_decode.h"

namespace ai
{
    class IDecoder
    {
    public:
        virtual ~IDecoder() {}
        virtual int Create(AIVdecCreateParams *params) = 0;
        virtual int Destroy() = 0;
        virtual int SendStream(const AIVdecStream *stream, int timeout_ms) = 0;
    };

    IDecoder *CreateDecoder();

} // namespace ai

#endif // AI_DECODE_IMPL_HPP_