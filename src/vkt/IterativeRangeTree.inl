// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

#include <vkt/ExecutionPolicy.hpp>

namespace vkt
{
    void IterativeRangeTree::update(Vec4f const* rgba, int size)
    {
        size_ = size;

        Base::resize(size_ + size_ - 1);

        Base::migrate();

        float* data = Base::data_;

        ExecutionPolicy ep = GetThreadExecutionPolicy();

        if (ep.device == ExecutionPolicy::Device::GPU)
        {
        }
        else
        {
            for (int i = 0; i < size; ++i)
            {
                data[i] = rgba[i].w;
            }

            int levelBegin = 0;
            int numIn = size;
            int numNodes = size;

            while (numIn >= 2)
            {
                for (int i = 0; i < numIn / 2; ++i)
                {
                    data[numNodes + i] = std::max(data[levelBegin + 2 * i],
                                                  data[levelBegin + 2 * i + 1]);
                }
                levelBegin += numIn;
                numIn /= 2;
                numNodes += numIn;
            }
        }
    }

    float* IterativeRangeTree::data()
    {
        Base::migrate();

        return Base::data_;
    }

    float const* IterativeRangeTree::data() const
    {
        const_cast<IterativeRangeTree*>(this)->migrate();

        return Base::data_;
    }

    int IterativeRangeTree::size() const
    {
        return size_;
    }

    IterativeRangeTreeView::IterativeRangeTreeView(float const* data, int size)
        : data_(data)
        , size_(size)
    {
    }

    VKT_FUNC
    float IterativeRangeTreeView::maxOpacity(Vec2f valueRange) const
    {
        float result = 0.f;

        int lo = floor(valueRange.x * (size_ - 1));
        int hi = ceil(valueRange.y * (size_ - 1));

        if ((lo & 1) == 1)
            result = max(result, data_[lo]);

        if ((hi & 1) == 0)
            result = max(result, data_[hi]);

        lo = (lo + 1) >> 1;
        hi = (hi - 1) >> 1;

        int off = size_;
        int numNodes = size_ / 2;

        while (lo <= hi)
        {
            if ((lo & 1) == 1)
                result = max(result, data_[off + lo]);

            if ((hi & 1) == 0)
                result = max(result, data_[off + hi]);

            lo = (lo + 1) >> 1;
            hi = (hi - 1) >> 1;

            off += numNodes;
            numNodes /= 2;
        }

        return result;
    }
} // vkt
