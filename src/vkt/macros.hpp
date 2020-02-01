// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/ExecutionPolicy.hpp>

#define VKT_CUDA_SAFE_CALL__(X) X                                               \

#define VKT_CALL__(FUNC, ...)                                                   \
    vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();                  \
                                                                                \
    if (ep.device == vkt::ExecutionPolicy::Device::CPU)                         \
    {                                                                           \
        if (ep.hostApi == vkt::ExecutionPolicy::HostAPI::Serial)                \
        {                                                                       \
            FUNC##_serial(__VA_ARGS__);                                         \
        }                                                                       \
    }                                                                           \
    else if (ep.device == vkt::ExecutionPolicy::Device::GPU)                    \
    {                                                                           \
        if (ep.deviceApi == vkt::ExecutionPolicy::DeviceAPI::CUDA)              \
        {                                                                       \
            FUNC##_cuda(__VA_ARGS__);                                           \
        }                                                                       \
    }                                                                           \
    else                                                                        \
    {                                                                           \
    }                                                                           \
