// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

#include <vkt/config.h>
#include <vkt/ExecutionPolicy.hpp>

#include "Logging.hpp"
#include "Timer.hpp"

#if VKT_HAVE_CUDA
#include "CudaTimer.hpp"
#endif

//-------------------------------------------------------------------------------------------------
// VKT_CALL_TIMER_
//

#define VKT_CALL_TIMER_(FUNC, ...)                                              \
    vkt::Timer timer;                                                           \
    char const* api[] = { "serial", "threads" };                                \
    FUNC(__VA_ARGS__);                                                          \
    VKT_LOG(vkt::logging::Level::Info)                                          \
            << "Device: CPU (" << api[(int)ep.hostApi] << "), algorithm: "      \
            << #FUNC                                                            \
            << ", time elapsed: "                                               \
            << timer.getElapsedSeconds()                                        \
            << " sec.";                                                         \

//-------------------------------------------------------------------------------------------------
// VKT_CALL_CUDA_TIMER_
//

#if VKT_HAVE_CUDA
#define VKT_CALL_CUDA_TIMER_(FUNC, ...)                                         \
    vkt::CudaTimer timer;                                                       \
    FUNC(__VA_ARGS__);                                                          \
    VKT_LOG(vkt::logging::Level::Info)                                          \
            << "Device: GPU (CUDA), algorithm: "                                \
            << #FUNC                                                            \
            << ", time elapsed: "                                               \
            << timer.getElapsedSeconds()                                        \
            << " sec.";
#else
#define VKT_CALL_CUDA_TIMER_(FUNC, ...)                                         \
    VKT_LOG(vkt::logging::Level::Error)                                         \
            << "When calling algorithm: "                                       \
            << #FUNC                                                            \
            << "CUDA backend unavailable.";
#endif

//-------------------------------------------------------------------------------------------------
// VKT_CALL_CUDA_ (w/o timer)
//

#if VKT_HAVE_CUDA
#define VKT_CALL_CUDA_(FUNC, ...)                                               \
    FUNC(__VA_ARGS__);
#else
#define VKT_CALL_CUDA_(FUNC, ...)                                               \
    VKT_LOG(vkt::logging::Level::Error)                                         \
            << "When calling algorithm: "                                       \
            << #FUNC                                                            \
            << "CUDA backend unavailable.";
#endif


//-------------------------------------------------------------------------------------------------
// LEGACY API - don't use anymore!!
//


//-------------------------------------------------------------------------------------------------
// VKT_LEGACY_CALL__
// Deprecated function call API, only used by legacy code!
//

#define VKT_LEGACY_CALL__(FUNC, ...)                                            \
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
            {                                                                   \
                VKT_CALL_CUDA_(FUNC##_cuda, __VA_ARGS__)                        \
            }                                                                   \
        }                                                                       \
    }                                                                           \
    else                                                                        \
    {                                                                           \
    }                                                                           \


//-------------------------------------------------------------------------------------------------
// Callable API
//

namespace vkt
{
    struct CallableBase
    {
        enum class API
        {
            Serial,
            //Threads,
            OpenMP,
            //TBB,
            CUDA,
            OpenCL,
        };

        CallableBase(const char* n) : name(n) {}

        virtual void call() = 0;

        const char* name = "";
    };

    template <typename ...Args>
    struct Callable : CallableBase
    {
        using Bound = decltype(std::bind(std::declval<std::function<void(Args...)>>(), std::declval<Args>()...));

        template <typename ...Brgs>
        Callable(const char* n, std::function<void(Args...)> func, Brgs... brgs)
            : CallableBase(n)
            , bound_(func, std::forward<Args>(brgs)...)
        {
        }

        void call()
        {
            return bound_();
        }

        // TODO: that's not used yet; intention is that one can specify
        // which other API is used as fallback, e.g., if OpenMP is not available,
        // resort to Serial (NOTE: _if_ we finally implement that, include some
        // assertions / runtime checks etc. to make sure that there are no circles!)
        void setFallBack(CallableBase* fallBack)
        {
            fallBack_ = fallBack;
        }

        Bound bound_;

        CallableBase* fallBack_ = nullptr;
    };

    template <typename ...Args>
    auto MakeCallable(const char* name, void (*func)(Args...), Args... args)
    {
        return Callable<Args...>(name, func, args...);
    }

    void Call(std::unordered_map<CallableBase::API, CallableBase*> callables);

} //vkt
