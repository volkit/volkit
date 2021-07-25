// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/Filter.hpp>
#include <vkt/linalg.hpp>

#include "macros.hpp"

namespace vkt
{
    class FilterView
    {
    public:
        VKT_FUNC FilterView() = default;
        VKT_FUNC FilterView(Filter& filter)
            : data_((float*)filter.getData())
            , dims_(filter.getDims())
        {
        }

        float operator()(int x, int y, int z) const
        {
            return data_[z * dims_.x * dims_.y + y * dims_.x + x];
        }

    private:
        float* data_;
        Vec3i dims_;
    };
}
