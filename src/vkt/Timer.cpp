// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "Timer.hpp"

//-------------------------------------------------------------------------------------------------
// CUDA timer
//

namespace vkt
{
    Timer::Timer()
        : start_(clock::now())
    {
    }

    void Timer::reset()
    {
        start_ = clock::now();
    }

    double Timer::getElapsedSeconds() const
    {
        return std::chrono::duration<double>(clock::now() - start_).count();
    }

} // vkt
