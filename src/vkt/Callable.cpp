// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "Callable.hpp"

namespace vkt
{
    bool tryCall(
            std::unordered_map<CallableBase::API, CallableBase*> callables,
            CallableBase::API api
            )
    {
        auto it = callables.find(api);
        if (it == callables.end())
        {
            return false;
        }
        else
        {
            vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
            if (ep.printPerformance)
            {
                if (ep.deviceApi == vkt::ExecutionPolicy::DeviceAPI::CUDA)
                {
#if VKT_HAVE_CUDA
                    vkt::CudaTimer timer;
                    it->second->call();
                    VKT_LOG(vkt::logging::Level::Info) << "Algorithm: "
                        << it->second->name << ", time elapsed: "
                        << timer.getElapsedSeconds() << " sec.";
#else
                    VKT_LOG(vkt::logging::Level::Error) << "When calling algorithm: "
                        << it->second->name << " CUDA backend unavailable.";
#endif
                }
                else
                {
                    vkt::Timer timer;
                    it->second->call();
                    VKT_LOG(vkt::logging::Level::Info) << "Algorithm: "
                        << it->second->name << ", time elapsed: "
                        << timer.getElapsedSeconds() << " sec.";
                }
            }
            else
            {
                it->second->call();
            }
            return true;
        }
    }

    void Call(std::unordered_map<CallableBase::API, CallableBase*> callables)
    {
        vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
        if (ep.device == vkt::ExecutionPolicy::Device::CPU)
        {
            if (ep.hostApi == vkt::ExecutionPolicy::HostAPI::Serial)
            {
                auto it = callables.find(CallableBase::API::Serial);

                if (!tryCall(callables, CallableBase::API::Serial))
                    VKT_LOG(vkt::logging::Level::Error) << "Not implemented for API: Serial\n";
            }
        }
    }
} // vkt
