// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cuda_runtime_api.h>

namespace vkt
{
    class CudaTimer
    {
    public:
        CudaTimer();
       ~CudaTimer();

        void reset();

        double getElapsedSeconds() const;

    private:
        cudaEvent_t start_;
        cudaEvent_t stop_;

    };

} // vkt
