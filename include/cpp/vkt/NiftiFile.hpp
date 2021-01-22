// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common.hpp"
#include "linalg.hpp"

namespace vkt
{
    class NiftiFile : public DataSource
    {
    public:
        NiftiFile(char const* fileName);
       ~NiftiFile();

        virtual std::size_t read(char* buf, std::size_t len);
        virtual std::size_t write(char const* buf, std::size_t len);
        virtual bool seek(std::size_t pos);
        virtual bool flush();
        virtual bool good() const;

        void setDims(Vec3i dims);

        Vec3i getDims();

        void setDataFormat(DataFormat dataFormat);

        DataFormat getDataFormat();

        void setVoxelMapping(Vec2f mapping);

        Vec2f getVoxelMapping();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

    };

} // vkt
