// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <vkt/CudaTimer.hpp>

#include <vkt/CudaTimer.h>

#include "macros.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
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


//-------------------------------------------------------------------------------------------------
// C API
//

struct vktCudaTimer_impl
{
    vkt::CudaTimer timer;
};

void vktCudaTimerCreate(vktCudaTimer* timer)
{
    assert(timer != nullptr);

    *timer = new vktCudaTimer_impl;
}

void vktCudaTimerDestroy(vktCudaTimer timer)
{
    delete timer;
}

void vktCudaTimerReset(vktCudaTimer timer)
{
    timer->timer.reset();
}

double vktCudaTimerGetElapsedSeconds(vktCudaTimer timer)
{
    return timer->timer.getElapsedSeconds();
}
