#pragma once

namespace vkt
{
    struct ExecutionPolicy
    {
        enum class Device
        {
            CPU,
            GPU,
            Unspecified,
        };

        enum class HostAPI
        {
            Serial,
            //OpenMP,
            //TBB,
        };

        enum class DeviceAPI
        {
            CUDA,
            //OpenCL,
        };

        Device device = Device::CPU;
        HostAPI hostApi = HostAPI::Serial;
        DeviceAPI deviceApi = DeviceAPI::CUDA;
    };

    /*!
     * @brief  set the execution of the current thread
     */
    void SetThreadExecutionPolicy(ExecutionPolicy policy);

    /*!
     * @brief  get the execution of the current thread
     */
    ExecutionPolicy GetThreadExecutionPolicy();

} // vkt
