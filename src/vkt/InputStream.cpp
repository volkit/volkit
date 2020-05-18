// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <vkt/InputStream.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/InputStream.h>
#include <vkt/StructuredVolume.h>

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
            int32_t lastZ,
            int32_t dstOffsetX,
            int32_t dstOffsetY,
            int32_t dstOffsetZ
            )
    {
        return NoError;
    }

    Error InputStream::readRange(
            StructuredVolume& dst,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        return NoError;
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
        int32_t lastZ,
        int32_t dstOffsetX,
        int32_t dstOffsetY,
        int32_t dstOffsetZ
        )
{
    /*stream->stream.readRange(
            volume->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ,
            dstOffsetX,
            dstOffsetY,
            dstOffsetZ
            );*/

    return vktNoError;
}

