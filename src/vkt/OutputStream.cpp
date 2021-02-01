// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <vkt/OutputStream.hpp>
#include <vkt/StructuredVolume.hpp>

#include "DataFormatInfo.hpp"
#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    OutputStream::OutputStream(DataSource& source)
        : dataSource_(source)
    {
    }

    Error OutputStream::write(StructuredVolume& volume)
    {
        if (!dataSource_.good())
            return InvalidDataSource;

        std::size_t len = dataSource_.write((char const*)volume.getData(), volume.getSizeInBytes());

        if (len != volume.getSizeInBytes())
            return WriteError;

        return NoError;
    }

    Error OutputStream::writeRange(
            StructuredVolume& dst,
            int32_t firstX,
            int32_t firstY,
            int32_t firstZ,
            int32_t lastX,
            int32_t lastY,
            int32_t lastZ
            )
    {
        uint16_t bpv = getSizeInBytes(dst.getDataFormat());
        std::size_t lineBytes = (lastX - firstX) * bpv;
        Vec3i dims = dst.getDims();

        std::size_t len = 0;

        for (int32_t z = firstZ; z != lastZ; ++z)
        {
            for (int32_t y = firstY; y != lastY; ++y)
            {
                std::size_t offset = (z * dims.x * std::size_t(dims.y) + y * dims.x) * bpv;
                len += dataSource_.write((char const*)dst.getData() + offset, lineBytes);
            }
        }

        if (len != (lastZ - firstZ) * (lastY - firstY) * lineBytes)
            return WriteError;

        return NoError;
    }

    Error OutputStream::writeRange(StructuredVolume& dst, Vec3i first, Vec3i last)
    {
        return writeRange(dst, first.x, first.y, first.z, last.x, last.y, last.z);
    }

    Error OutputStream::seek(std::size_t pos)
    {
        if (dataSource_.seek(pos))
            return NoError;

        // TODO: more specific
        return InvalidValue;
    }

    Error OutputStream::flush()
    {
        if (dataSource_.flush())
            return NoError;

        // TODO: more specific
        return InvalidValue;
    }

} // vkt
