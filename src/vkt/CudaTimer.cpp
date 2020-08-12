// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "CudaTimer.hpp"
#include "macros.hpp"

//-------------------------------------------------------------------------------------------------
// CUDA timer
//

namespace vkt
{
    CudaTimer::CudaTimer()
    {
        VKT_CUDA_SAFE_CALL__(cudaEventCreate(&start_));
        VKT_CUDA_SAFE_CALL__(cudaEventCreate(&stop_));

        reset();
    }

    CudaTimer::~CudaTimer()
    {
        VKT_CUDA_SAFE_CALL__(cudaEventDestroy(start_));
        VKT_CUDA_SAFE_CALL__(cudaEventDestroy(stop_));
    }

    void CudaTimer::reset()
    {
        VKT_CUDA_SAFE_CALL__(cudaEventRecord(start_));
    }

    double CudaTimer::getElapsedSeconds() const
    {
        VKT_CUDA_SAFE_CALL__(cudaEventRecord(stop_));
        VKT_CUDA_SAFE_CALL__(cudaEventSynchronize(stop_));
        float ms = 0.f;
        VKT_CUDA_SAFE_CALL__(cudaEventElapsedTime(&ms, start_, stop_));
        return static_cast<double>(ms) / 1000.;
    }

} // vkt
