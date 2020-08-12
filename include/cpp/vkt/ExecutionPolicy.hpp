// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.hpp"

/*! @file  ExecutionPolicy.hpp
 * @brief  Utilities to query or modify the current thread execution policy
 *
 * Volkit uses a deferred API where the way that computations are performed
 * (e.g., on the CPU or on the GPU, preferred parallel API, etc.) is specified
 * in advance before invoking an algorithm and on a per thread basis. With each
 * application thread there is associated a @ref vkt::ExecutionPolicy object; the
 * user can modify that object to influence how ensuing computations are
 * performed.
 *
 * The following example shows how to set the execution policy to `GPU` for the
 * current thread:
 * ```
 * // Current execution policy is "CPU"
 * vkt::StructuredVolume volume;
 *
 * //...
 * 
 * vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
 * ep.device = vkt::ExecutionPolicy::Device::GPU;
 * vkt::SetThreadExecutionPolicy(ep);
 *
 * // Execution policy is now "GPU"
 * // Data is still on the CPU though as the API is deferred
 *
 * // The Fill algorithm will determine that the execution policy
 * // when the volume was created was "CPU" and will now "migrate"
 * // the volume so that the data is on the GPU before performing
 * // the fill operation
 * vkt::Fill(volume, 0.f);
 * ```
 * Therefore, setting the execution policy is cheap, but the next algorithm or
 * function call might involve data migration and thus a copy over PCIe. Note
 * that setting and immediately resetting the execution policy results in a
 * noop, i.e., algorithms will not detect changes to the execution policy but
 * only the immediate policies specified for current and previous usage.
 */
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

    /*! @brief  Set the execution policy of the current thread
     *
     * Use this function to set the thread execution policy. Setting the execution
     * policy affects how the ensuing algorithms and operations are executed.
     * @see GetThreadExecutionPolicy()
     */
    VKTAPI void SetThreadExecutionPolicy(ExecutionPolicy policy);

    /*! @brief  Get the execution policy of the current thread
     *
     * Each application / user thread has its own execution policy. Use this
     * function to query the execution policy of the current thread. Note that
     * execution policies are not inherited by child threads. This function can
     * be used to query the execution policy in the parent thread and then set
     * it via @ref SetThreadExecutionPolicy() in the child thread.
     * @see SetThreadExecutionPolicy()
     */
    VKTAPI ExecutionPolicy GetThreadExecutionPolicy();

} // vkt
