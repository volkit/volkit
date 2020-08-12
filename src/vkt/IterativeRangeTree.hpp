// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/linalg.hpp>

#include "macros.hpp"

namespace vkt
{
    class IterativeRangeTree : public ManagedBuffer<float>
    {
    public:
        typedef ManagedBuffer<float> Base;

        friend class IterativeRangeTreeView;

        void update(Vec4f const* rgba, int size);

        float* data();

        float const* data() const;

        int size() const;

    private:
        int size_;
    };


    class IterativeRangeTreeView
    {
    public:
        IterativeRangeTreeView() = default;

        IterativeRangeTreeView(float const* data, int size);

        VKT_FUNC
        float maxOpacity(Vec2f valueRange) const;

    private:
        float const* data_;
        int size_;

    };

} // vkt

#include "IterativeRangeTree.inl"
