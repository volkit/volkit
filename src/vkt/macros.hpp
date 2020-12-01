// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/ExecutionPolicy.hpp>

#include "CudaTimer.hpp"
#include "Logging.hpp"
#include "Timer.hpp"

#define VKT_CUDA_SAFE_CALL__(X) X                                               \

//-------------------------------------------------------------------------------------------------
// VKT_CALL_TIMER_
//

#define VKT_CALL_TIMER_(FUNC, ...)                                              \
    vkt::Timer timer;                                                           \
    FUNC(__VA_ARGS__);                                                          \
    VKT_LOG(vkt::logging::Level::Info)                                          \
            << "Device: CPU (serial), algorithm: "                              \
            << #FUNC                                                            \
            << ", time elapsed: "                                               \
            << timer.getElapsedSeconds()                                        \
            << " sec.";                                                         \

//-------------------------------------------------------------------------------------------------
// VKT_CALL_CUDA_TIMER_
//

#define VKT_CALL_CUDA_TIMER_(FUNC, ...)                                         \
    vkt::CudaTimer timer;                                                       \
    FUNC(__VA_ARGS__);                                                          \
    VKT_LOG(vkt::logging::Level::Info)                                          \
            << "Device: GPU (CUDA), algorithm: "                                \
            << #FUNC                                                            \
            << ", time elapsed: "                                               \
            << timer.getElapsedSeconds()                                        \
            << " sec.";                                                         \

//-------------------------------------------------------------------------------------------------
// VKT_CALL__
//

#define VKT_CALL__(FUNC, ...)                                                   \
    vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();                  \
                                                                                \
    if (ep.device == vkt::ExecutionPolicy::Device::CPU)                         \
    {                                                                           \
        if (ep.hostApi == vkt::ExecutionPolicy::HostAPI::Serial)                \
        {                                                                       \
            if (ep.printPerformance)                                            \
            {                                                                   \
                VKT_CALL_TIMER_(FUNC##_serial, __VA_ARGS__)                     \
            }                                                                   \
            else                                                                \
                FUNC##_serial(__VA_ARGS__);                                     \
        }                                                                       \
    }                                                                           \
    else if (ep.device == vkt::ExecutionPolicy::Device::GPU)                    \
    {                                                                           \
        if (ep.deviceApi == vkt::ExecutionPolicy::DeviceAPI::CUDA)              \
        {                                                                       \
            if (ep.printPerformance)                                            \
            {                                                                   \
                VKT_CALL_CUDA_TIMER_(FUNC##_cuda, __VA_ARGS__)                  \
            }                                                                   \
            else                                                                \
                FUNC##_cuda(__VA_ARGS__);                                       \
        }                                                                       \
    }                                                                           \
    else                                                                        \
    {                                                                           \
    }                                                                           \

//-------------------------------------------------------------------------------------------------
// CUDA host/device
//

#ifdef __CUDACC__
#define VKT_FUNC __host__ __device__
#else
#define VKT_FUNC
#endif
