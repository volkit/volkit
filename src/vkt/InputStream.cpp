// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <vkt/InputStream.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/InputStream.h>
#include <vkt/StructuredVolume.h>

#include "DataFormatInfo.hpp"
#include "InputStream_impl.hpp"
#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

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

    Error InputStream::readRange(
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
                len += dataSource_.read((char*)dst.getData() + offset, lineBytes);
            }
        }

        if (len != (lastZ - firstZ) * (lastY - firstY) * lineBytes)
            return ReadError;

        return NoError;
    }

    Error InputStream::readRange(StructuredVolume& dst, Vec3i first, Vec3i last)
    {
        return readRange(dst, first.x, first.y, first.z, last.x, last.y, last.z);
    }

    Error InputStream::seek(std::size_t pos)
    {
        if (dataSource_.seek(pos))
            return NoError;

        // TODO: more specific
        return InvalidValue;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

void vktInputStreamCreate(vktInputStream* stream, vktDataSource source)
{
    assert(stream != nullptr);

    *stream = new vktInputStream_impl(source);
}

void vktInputStreamDestroy(vktInputStream stream)
{
    delete stream;
}

vktError vktInputStreamReadSV(vktInputStream stream, vktStructuredVolume volume)
{
    stream->stream.read(volume->volume);

    return vktNoError;
}

vktError vktInputStreamReadRangeSV(
        vktInputStream stream,
        vktStructuredVolume volume,
        int32_t firstX,
        int32_t firstY,
        int32_t firstZ,
        int32_t lastX,
        int32_t lastY,
        int32_t lastZ
        )
{
    stream->stream.readRange(
            volume->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ
            );

    return vktNoError;
}

vktError vktInputStreamSeek(vktInputStream stream, size_t pos)
{
    vkt::Error err = stream->stream.seek(pos);

    if (err == vkt::NoError)
        return vktNoError;

    // TODO: more specific
    return vktInvalidValue;
}
