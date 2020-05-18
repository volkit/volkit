// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <cstring>

#include <vkt/config.h>

#if VKT_HAVE_VIRVO
#include <virvo/vvfileio.h>
#include <virvo/vvvoldesc.h>
#endif

#include <vkt/VirvoFile.hpp>

#include <vkt/VirvoFile.h>

namespace vkt
{
    struct VirvoFile::Impl
    {
        Impl(char const* fileName)
#if VKT_HAVE_VIRVO
            : vd(fileName)
#endif
        {
        }
#if VKT_HAVE_VIRVO
        vvVolDesc vd;
#endif

        bool loaded() const
        {
#if VKT_HAVE_VIRVO
            return vd.vox[0] * vd.vox[1] * vd.vox[2] > 0;
#else
            return false;
#endif
        }

        bool load()
        {
#if VKT_HAVE_VIRVO
            vvFileIO fileio;
            return fileio.loadVolumeData(&vd) == vvFileIO::OK;

#else
            return false;
#endif
        }

        std::size_t readOffset = 0;
    };

    VirvoFile::VirvoFile(char const* fileName)
        : impl_(new Impl(fileName))
    {
    }

    VirvoFile::~VirvoFile()
    {
    }

    std::size_t VirvoFile::read(char* buf, std::size_t len)
    {
#if VKT_HAVE_VIRVO
        if (!impl_->loaded() && !impl_->load())
            return 0;

        uint8_t* raw = impl_->vd.getRaw();

        // TODO: error / bounds checking
        std::memcpy(buf, raw + impl_->readOffset, len);

        impl_->readOffset += len;

        return len;
#else
        return 0;
#endif
    }

    bool VirvoFile::good() const
    {
#if VKT_HAVE_VIRVO
        return true; // TODO (check if exists / can be created??)
#else
        return false;
#endif
    }

    Vec3i VirvoFile::getDims()
    {
#if VKT_HAVE_VIRVO
        if (!impl_->loaded() && !impl_->load())
            return { 0, 0, 0 };

        return {
            (int)impl_->vd.vox[0],
            (int)impl_->vd.vox[1],
            (int)impl_->vd.vox[2]
            };
#else
        return {};
#endif
    }

    uint16_t VirvoFile::getBytesPerVoxel()
    {
#if VKT_HAVE_VIRVO
        if (!impl_->loaded() && !impl_->load())
            return 0;

        return (uint16_t)impl_->vd.bpc;
#else
        return 0;
#endif
    }

} // vkt
