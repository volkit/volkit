// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/VolumeFile.hpp>

#include "DataSource_impl.hpp"

struct vktVolumeFile_impl
{
    vktVolumeFile_impl(char const* fileName, vktOpenMode om)
        : base(new vktDataSource_impl)
    {
        base->source = new vkt::VolumeFile(fileName, (vkt::OpenMode)om);
    }

   ~vktVolumeFile_impl()
    {
        delete base->source;
        delete base;
    }

    vktDataSource base;
};
