#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>

#include <vkt/ExecutionPolicy.hpp>

static void print(vkt::ExecutionPolicy ep)
{
    std::string device = ep.device == vkt::ExecutionPolicy::Device::CPU
                                ? "CPU" : "GPU";

    std::string hostApi = ep.hostApi == vkt::ExecutionPolicy::HostAPI::Serial
                                ? "Serial" : "???";

    std::string deviceApi = ep.deviceApi == vkt::ExecutionPolicy::DeviceAPI::CUDA
                                ? "CUDA" : "???";

    std::ostringstream str;
    str << "ExecutionPolicy thread " << std::this_thread::get_id() << '\n';
    str << "device .....: " << device << '\n';
    str << "hostApi ....: " << hostApi << '\n';
    str << "deviceApi ..: " << deviceApi << '\n';
    std::cout << '\n' << str.str() << '\n';
}

void threadFunc()
{
    // Create a default execution policy
    vkt::ExecutionPolicy ep;
    print(ep);

    // Set this as policy for this thread
    // (this will have no effect)
    vkt::SetThreadExecutionPolicy(ep);

    // Change to GPU
    ep.device = vkt::ExecutionPolicy::Device::GPU;
    vkt::SetThreadExecutionPolicy(ep);

    // Changes were applied
    ep = vkt::GetThreadExecutionPolicy();
    print(ep);
}

int main()
{
    // Default execution policy (CPU)
    vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
    print(ep);

    // Change execution policy in main thread
    ep.device = vkt::ExecutionPolicy::Device::GPU;
    vkt::SetThreadExecutionPolicy(ep);

    // Changes were applied
    ep = vkt::GetThreadExecutionPolicy();
    print(ep);

    // New thread
    std::thread t(threadFunc);
    t.join();

    // Execution policy in main thread remains unchanged
    ep = vkt::GetThreadExecutionPolicy();
    print(ep);
}
