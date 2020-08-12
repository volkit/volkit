// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <chrono>

namespace vkt
{
    class Timer
    {
    public:
        typedef std::chrono::high_resolution_clock clock;
        typedef clock::time_point time_point;
        typedef clock::duration duration;

        Timer();

        void reset();

        double getElapsedSeconds() const;

    private:
        time_point start_;

    };

} // vkt
