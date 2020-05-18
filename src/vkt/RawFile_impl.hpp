// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdio>

#include <vkt/RawFile.hpp>

#include "DataSource_impl.hpp"

struct vktRawFile_impl
{
    vktRawFile_impl(char const* fileName, char const* mode)
        : base(new vktDataSource_impl)
    {
        base->source = new vkt::RawFile(fileName, mode);
    }

    vktRawFile_impl(FILE* fd)
        : base(new vktDataSource_impl)
    {
        base->source = new vkt::RawFile(fd);
    }

   ~vktRawFile_impl()
    {
        delete base->source;
        delete base;
    }

    vktDataSource base;
};
