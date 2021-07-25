// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>

#include <vkt/ManagedBuffer.hpp>

#include "common.hpp"
#include "linalg.hpp"

namespace vkt
{
    /*! @class StructuredVolume
     * @brief  Managed structured volume class
     *
     * This class represents a structured volume in memory. The volume is
     * managed, i.e., where the actual data is stored (e.g., on the CPU or on
     * the GPU) is determined by the internal memory management layer. This can
     * be influenced by setting the active thread's execution policy via @ref
     * vkt::SetThreadExecutionPolicy(). When the execution policy was changed
     * so that the volume data should be present on another device, the
     * algorithm accessing the data next will initiate for the volume to be
     * migrated before the respective operation is performed. This happens in a
     * deferred fashion.
     *
     * Structured volumes have 3D dimensions, store the data format, the
     * distance between voxels in x,y, and z direction, as well as a linear
     * mapping from the minimum and maximum data value to the floating point
     * range `[lo..hi]`.
     */
    class StructuredVolume : public ManagedBuffer<uint8_t>
    {
    public:
        constexpr static uint8_t GetMaxBytesPerVoxel() { return 8; }
    
    public:
        StructuredVolume();
        StructuredVolume(
                int32_t dimX,
                int32_t dimY,
                int32_t dimZ,
                DataFormat dataFormat,
                float distX = 1.f,
                float distY = 1.f,
                float distZ = 1.f,
                float mappingLo = 0.f,
                float mappingHi = 1.f
                );
        StructuredVolume(StructuredVolume& rhs) = default;
        StructuredVolume(StructuredVolume&& rhs) = default;

        StructuredVolume& operator=(StructuredVolume& rhs) = default;
        StructuredVolume& operator=(StructuredVolume&& rhs) = default;

        void setDims(int32_t dimX, int32_t dimY, int32_t dimZ);
        void getDims(int32_t& dimX, int32_t& dimY, int32_t& dimZ);

        void setDims(Vec3i dims);
        Vec3i getDims() const;

        void setDataFormat(DataFormat dataFormat);
        DataFormat getDataFormat() const;

        //! Set cell distance (dflt: (1.,1.,1.))
        void setDist(float distX, float distY, float distZ);

        //! Get cell distance as three floats
        void getDist(float& distX, float& distY, float& distZ);

        //! Set cell distance (dflt: (1.,1.,1.))
        void setDist(Vec3f dist);

        //! Get cell distance as Vec3f
        Vec3f getDist() const;

        //! Set linear mapping from internal to float (dflt: (0.,1.))
        void setVoxelMapping(float lo, float hi);

        //! Get linear mapping from internal to float as two floats
        void getVoxelMapping(float& lo, float& hi);

        //! Set linear mapping from internal to float (dflt: (0.,1.))
        void setVoxelMapping(Vec2f mapping);

        //! Get linear mapping from internal to float as Vec2f
        Vec2f getVoxelMapping() const;

        //! Set the offset of the minimum corner in world space (dflt: (0,0,0))
        void setWorldOrigin(Vec3f worldOrigin);

        //! Get the offset of the minimum corner in world space
        Vec3f getWorldOrigin() const;

        //! Get the interpolation domain (cell bounds + halo) in world space
        Box3f getDomainBounds() const;

        //! Get the cell's bounds in world space
        Box3f getWorldBounds() const;

        //! Get a raw pointer to the internal data
        uint8_t* getData();

        float sampleLinear(int32_t x, int32_t y, int32_t z);

        void setValue(int32_t x, int32_t y, int32_t z, float value);
        void getValue(int32_t x, int32_t y, int32_t z, float& value);
        float getValue(int32_t x, int32_t y, int32_t z);

        void setValue(Vec3i index, float value);
        void getValue(Vec3i index, float& value);
        float getValue(Vec3i index);

        void setBytes(int32_t x, int32_t y, int32_t z, uint8_t const* data);
        void getBytes(int32_t x, int32_t y, int32_t z, uint8_t* data);

        void setBytes(Vec3i index, uint8_t const* data);
        void getBytes(Vec3i index, uint8_t* data);

        uint8_t getBytesPerVoxel() const;
        std::size_t getSizeInBytes() const;

    private:
        Vec3i dims_;
        DataFormat dataFormat_;
        Vec3f dist_;
        Vec2f voxelMapping_;
        Vec3f origin_;
        /* we currently *don't* expose the halo size to the user but
         set it (hardcoded) to .5f,.5f,.5f in the ctor. The halo size _is_
         exposed though through the getDomain() function, where the domain is
         just worldBounds+(-haloSize,+haloSize). We might make this editable
         later if we support higher order interpolation, or, (more likely) more
         accurate interpolation for AMR data */
        Vec3f haloSize_;

        std::size_t linearIndex(int32_t x, int32_t y, int32_t z) const;
        std::size_t linearIndex(Vec3i index) const;
    };
} // vkt
