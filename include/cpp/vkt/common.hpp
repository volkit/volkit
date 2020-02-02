// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>

/*! VKTAPI */
#define VKTAPI

namespace vkt
{
    /*! Boolean */
    typedef uint8_t Bool;

    /*! Error constants */
    enum Error
    {
        InvalidValue      = -1,
        NoError           =  0,
        InvalidDataSource =  1,
        ReadError         =  2,
    };

    enum class ColorFormat
    {
        Unspecified,

        R8,
        RG8,
        RGB8,
        RGBA8,
        R16UI,
        RG16UI,
        RGB16UI,
        RGBA16UI,
        R32UI,
        RG32UI,
        RGB32UI,
        RGBA32UI,
        R32F,
        RG32F,
        RGB32F,
        RGBA32F,

        // Keep last!
        Count,
    };


    /*!
     * @brief  Data source base class for file I/O
     */
    class DataSource
    {
    public:
        virtual ~DataSource() {}
        virtual std::size_t read(char* buf, std::size_t len) = 0;
        virtual bool good() const = 0;

    };

} // vkt
