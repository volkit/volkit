// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <climits>
#include <cstddef>
#include <cstring>
#include <vector>

#include <vkt/config.h>

#if VKT_HAVE_NIFTI
#include <nifti1_io.h>
#endif

#include <vkt/NiftiFile.hpp>

//#include <vkt/NiftiFile.h>

namespace vkt
{
    struct NiftiFile::Impl
    {
#if VKT_HAVE_NIFTI
        nifti_image* header = nullptr;
        nifti_image* dataSection = nullptr;
        std::size_t readOffset = 0;
#endif
    };

    NiftiFile::NiftiFile(char const* fileName)
        : impl_(new Impl)
    {
#if VKT_HAVE_NIFTI
        impl_->header = nifti_image_read(fileName, 0);
        impl_->dataSection = nifti_image_read(fileName, 1);
#endif
    }

    NiftiFile::~NiftiFile()
    {
#if VKT_HAVE_NIFTI
        nifti_image_free(impl_->dataSection);
        nifti_image_free(impl_->header);
#endif
    }

    std::size_t NiftiFile::read(char* buf, std::size_t len)
    {
#if VKT_HAVE_NIFTI
        if (!good())
            return 0;

        // TODO: error / bounds checking
        std::memcpy(
            buf,
            static_cast<uint8_t*>(impl_->dataSection->data) + impl_->readOffset,
            len
            );

        // Remap to internal data type
        if (impl_->header->datatype == NIFTI_TYPE_INT16)
        {
            uint16_t bpv = getBytesPerVoxel();

            for (std::size_t i = 0; i < len; i += bpv)
            {
                char* bytes = buf + i;
                int32_t voxel = static_cast<int32_t>(*reinterpret_cast<int16_t*>(bytes));
                voxel -= SHRT_MIN;
                *reinterpret_cast<uint16_t*>(bytes) = voxel;
            }
        }

        impl_->readOffset += len;

        return len;
#else
        return 0;
#endif
    }

    bool NiftiFile::good() const
    {
#if VKT_HAVE_NIFTI
        return impl_->dataSection;
#else
        return false;
#endif
    }

    Vec3i NiftiFile::getDims()
    {
#if VKT_HAVE_NIFTI
        return {
            impl_->header->nx,
            impl_->header->ny,
            impl_->header->nz
            };
#else
        return {};
#endif
    }

    uint16_t NiftiFile::getBytesPerVoxel()
    {
#if VKT_HAVE_NIFTI
        switch (impl_->header->datatype)
        {
        case NIFTI_TYPE_INT8:
        case NIFTI_TYPE_UINT8:
            return 1;

        case NIFTI_TYPE_INT16:
        case NIFTI_TYPE_UINT16:
            return 2;

        case NIFTI_TYPE_INT32:
        case NIFTI_TYPE_UINT32:
        case NIFTI_TYPE_FLOAT32:
            return 4;

        default:
            // unsupported
            return 0;
        }
#else
        return 0;
#endif
    }

    Vec2f NiftiFile::getVoxelMapping()
    {
#if VKT_HAVE_NIFTI
        float lo = 0.f;
        float hi = 1.f;

        switch (impl_->header->datatype)
        {
        case NIFTI_TYPE_INT16:
            lo = SHRT_MIN;
            hi = SHRT_MAX;
            break;
        }

        return {
            lo * impl_->header->scl_slope + impl_->header->scl_inter,
            hi * impl_->header->scl_slope + impl_->header->scl_inter
            };
#else
        return { };
#endif
    }

} // vkt
