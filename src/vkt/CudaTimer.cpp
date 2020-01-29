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

vktError vktCudaTimerCreate(vktCudaTimer* timer)
{
    if (timer == nullptr)
    {
        return VKT_INVALID_VALUE;
    }

    *timer = new vktCudaTimer_impl;

    return VKT_NO_ERROR;
}

vktError vktCudaTimerDestroy(vktCudaTimer timer)
{
    if (timer == nullptr)
    {
        return VKT_INVALID_VALUE;
    }

    delete timer;

    return VKT_NO_ERROR;
}

vktError vktCudaTimerReset(vktCudaTimer timer)
{
    if (timer == nullptr)
    {
        return VKT_INVALID_VALUE;
    }

    timer->timer.reset();

    return VKT_NO_ERROR;
}

vktError vktCudaTimerGetElapsedSeconds(vktCudaTimer timer, double* seconds)
{
    if (timer == nullptr)
    {
        return VKT_INVALID_VALUE;
    }

    *seconds = timer->timer.getElapsedSeconds();

    return VKT_NO_ERROR;
}
