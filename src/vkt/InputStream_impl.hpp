// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/InputStream.hpp>

#include "RawFile_impl.hpp"

struct vktInputStream_impl
{
    vktInputStream_impl(vktRawFile_impl* file)
        : stream(file->file)
    {
    }

    vkt::InputStream stream;
};
