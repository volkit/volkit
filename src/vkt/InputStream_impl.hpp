// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/InputStream.hpp>

#include "DataSource_impl.hpp"

struct vktInputStream_impl
{
    vktInputStream_impl(vktDataSource_impl* source)
        : stream(*source->source)
    {
    }

    vkt::InputStream stream;
};
