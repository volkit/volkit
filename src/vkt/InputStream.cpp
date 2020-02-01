// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/InputStream.hpp>
#include <vkt/StructuredVolume.hpp>

#include "StructuredVolume_impl.hpp"

namespace vkt
{
    InputStream::InputStream(DataSource& source)
        : dataSource_(source)
    {
    }

    Error InputStream::read(StructuredVolume& volume)
    {
        if (!dataSource_.good())
            return InvalidDataSource;

        std::size_t len = dataSource_.read((char*)volume.getData(), volume.getSizeInBytes());

        if (len != volume.getSizeInBytes())
            return ReadError;

        return NoError;
    }

} // vkt
