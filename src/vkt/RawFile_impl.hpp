// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdio>

#include <vkt/RawFile.hpp>

struct vktRawFile_impl
{
    vktRawFile_impl(char const* fileName, char const* mode)
        : file(fileName, mode)
    {
    }

    vktRawFile_impl(FILE* fd)
        : file(fd)
    {
    }

    vkt::RawFile file;
};
