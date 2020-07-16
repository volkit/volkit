// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstring>
#include <cstddef>

#include <cuda_runtime.h>

#include "linalg.hpp"
#include "macros.hpp"
#include "Memory_cuda.hpp"

namespace vkt
{
    __global__ void MemsetRange_kernel(
            void* dst,
            void const* src,
            std::size_t srcSize,
            std::size_t numElem
            )
    {
        std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;

        if (i >= numElem)
            return;

        memcpy((char*)dst + i * srcSize, src, srcSize);
    }

    void MemsetRange_cuda(
            void* dst,
            void const* src,
            std::size_t dstSize,
            std::size_t srcSize
            )
    {
        std::size_t numElem = dstSize / srcSize;

        dim3 blockSize(1024);
        dim3 gridSize(div_up((unsigned)numElem, blockSize.x));

        void* d_src = nullptr;
        VKT_CUDA_SAFE_CALL__(cudaMalloc(&d_src, srcSize));
        VKT_CUDA_SAFE_CALL__(cudaMemcpy(d_src, src, srcSize, cudaMemcpyHostToDevice));

        MemsetRange_kernel<<<gridSize, blockSize>>>(dst, d_src, srcSize, numElem);

        VKT_CUDA_SAFE_CALL__(cudaFree(d_src));
    }
}
