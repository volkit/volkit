// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>

/*! VKTAPI */
#define VKTAPI

namespace vkt
{
    /*! Error constants */
    enum Error
    {
        InvalidValue      = -1,
        NoError           =  0,
        InvalidDataSource =  1,
        ReadError         =  2,
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
