#pragma once

#include <vkt/config.h>

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    // Unary op
    void TransformRange_cuda(
            StructuredVolume& volume,
            vec3i first,
            vec3i last,
            TransformUnaryOp unaryOp
            )
#if 0//VKT_HAVE_CUDA
    ;
#else
    {}
#endif

    // Binary op
    void TransformRange_cuda(
            StructuredVolume& volume1,
            StructuredVolume& volume2,
            vec3i first,
            vec3i last,
            TransformBinaryOp binaryOp
            )
#if 0//VKT_HAVE_CUDA
    ;
#else
    {}
#endif
} // vkt
