// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdio>
#include <cstddef>

#include "common.hpp"

namespace vkt
{
    class RawFile : public DataSource
    {
    public:
        RawFile(char const* fileName, char const* mode);
        RawFile(FILE* file);
       ~RawFile();

        virtual std::size_t read(char* buf, std::size_t len);
        virtual bool good() const;

        // TODO: access function getDims(), guess those from fileName,
        //       return -1 if unsuccessful

    public:
        char const* fileName_ = 0;
        char const* mode_ = 0;
        FILE* file_ = 0;

    };

} // vkt
