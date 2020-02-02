// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <vkt/LookupTable.hpp>

#include <vkt/LookupTable.h>

#include "ColorFormatInfo.hpp"
#include "LookupTable_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    LookupTable::LookupTable(
            int32_t dimX,
            int32_t dimY,
            int32_t dimZ,
            ColorFormat format
            )
        : ManagedBuffer(dimX * size_t(dimY) * dimZ * ColorFormatInfoTable[(int)format].size)
        , dims_(dimX, dimY, dimZ)
        , format_(format)
    {
    }

    void LookupTable::setDims(int32_t dimX, int32_t dimY, int32_t dimZ)
    {
        setDims(Vec3i(dimX, dimY, dimZ));
    }

    void LookupTable::getDims(int32_t& dimX, int32_t& dimY, int32_t& dimZ)
    {
        dimX = dims_.x;
        dimY = dims_.y;
        dimZ = dims_.z;
    }

    void LookupTable::setDims(Vec3i dims)
    {
        dims_ = dims;
        std::size_t newSize = getSizeInBytes();

        resize(newSize);
    }

    Vec3i LookupTable::getDims() const
    {
        return dims_;
    }

    void LookupTable::setColorFormat(ColorFormat format)
    {
        format_ = format;
        std::size_t newSize = getSizeInBytes();

        resize(newSize);
    }

    ColorFormat LookupTable::getColorFormat() const
    {
        return format_;
    }

    uint8_t* LookupTable::getData()
    {
        migrate();

        return ManagedBuffer::data_;
    }

    std::size_t LookupTable::getSizeInBytes() const
    {
        return dims_.x * size_t(dims_.y) * dims_.z * ColorFormatInfoTable[(int)format_].size;
    }


} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

void vktLookupTableCreate(
        vktLookupTable* lut,
        int32_t dimX,
        int32_t dimY,
        int32_t dimZ,
        vktColorFormat format
        )
{
    assert(lut != nullptr);

    *lut = new vktLookupTable_impl(dimX, dimY, dimZ, (vkt::ColorFormat)format);
}

void vktLookupTableDestroy(vktLookupTable lut)
{
    delete lut;
}

void vktLookupTableSetDims3i(
        vktLookupTable lut,
        int32_t dimX,
        int32_t dimY,
        int32_t dimZ
        )
{
    lut->lut.setDims(dimX, dimY, dimZ);
}

void vktLookupTableGetDims3i(
        vktLookupTable lut,
        int32_t* dimX,
        int32_t* dimY,
        int32_t* dimZ
        )
{
    lut->lut.getDims(*dimX, *dimY, *dimZ);
}

void vktLookupTableSetDims3iv(vktLookupTable lut, vktVec3i_t dims)
{
    lut->lut.setDims(dims.x, dims.y, dims.z);
}

vktVec3i_t vktLookupTableGetDims3iv(vktLookupTable lut)
{
    vkt::Vec3i dims = lut->lut.getDims();

    return { dims.x, dims.y, dims.z };
}

void vktLookupTableSetColorFormat(vktLookupTable lut, vktColorFormat format)
{
    lut->lut.setColorFormat((vkt::ColorFormat)format);
}

vktColorFormat vktLookupTableGetColorFormat(vktLookupTable lut)
{
    return (vktColorFormat)lut->lut.getColorFormat();
}

uint8_t* vktLookupTableGetData(vktLookupTable lut)
{
    return lut->lut.getData();
}

size_t vktLookupTableGetSizeInBytes(vktLookupTable lut)
{
    return lut->lut.getSizeInBytes();
}

vktResourceHandle vktLookupTableGetResourceHandle(vktLookupTable lut)
{
    return (vktResourceHandle)lut->lut.getResourceHandle();
}

void vktLookupTableMigrate(vktLookupTable lut)
{
    lut->lut.migrate();
}
