// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdio>
#include <cstddef>
#include <cstdint>

#include "common.hpp"
#include "linalg.hpp"

namespace vkt
{
    class RawFile : public DataSource
    {
    public:
        RawFile(char const* fileName, char const* mode);
        RawFile(FILE* file);
       ~RawFile();

        virtual std::size_t read(char* buf, std::size_t len);
        virtual std::size_t write(char const* buf, std::size_t len);
        virtual bool seek(std::size_t pos);
        virtual bool flush();
        virtual bool good() const;

        /*!
         * @brief  Set structured volume dimensions
         */
        void setDims(Vec3i dims);

        /*!
         * @brief  Structured volume dimensions parsed from file name,
         *         0 if not successful
         */
        Vec3i getDims() const;

        /*!
         * @brief  Set structured volume data format
         */
        void setDataFormat(DataFormat dataFormat);

        /*!
         * @brief  Get structured volume data format
         */
        DataFormat getDataFormat() const;

    private:
        char const* fileName_ = 0;
        char const* mode_ = 0;
        FILE* file_ = 0;

        Vec3i dims_ = { 0, 0, 0 };
        DataFormat dataFormat_ = DataFormat::UInt8;

    };

} // vkt
