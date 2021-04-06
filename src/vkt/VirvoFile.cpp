// This file is distributed under the MIT license.
// See the LICENSE file for details.


#include <cstddef>
#include <cstring>

#include <vkt/config.h>

#if VKT_HAVE_VIRVO
#include <virvo/vvfileio.h>
#include <virvo/vvtransfunc.h>
#include <virvo/vvvoldesc.h>
#endif

#include <vkt/VirvoFile.hpp>

#include <vkt/VirvoFile.h>

#include "DataFormatInfo.hpp"

namespace vkt
{
    struct VirvoFile::Impl
    {
        Impl(char const* fileName)
#if VKT_HAVE_VIRVO
            : vd(new vvVolDesc(fileName))
#endif
        {
        }
#if VKT_HAVE_VIRVO
        vvVolDesc* vd;
#endif

       ~Impl()
        {
            if (needFlush)
                flush();

#if VKT_HAVE_VIRVO
            delete vd;
#endif
        }

        bool initialized() const
        {
#if VKT_HAVE_VIRVO
            return vd->vox[0] * vd->vox[1] * vd->vox[2] > 0;
#else
            return false;
#endif
        }

        bool load()
        {
#if VKT_HAVE_VIRVO
            vvFileIO fileio;
            return fileio.loadVolumeData(vd) == vvFileIO::OK;

#else
            return false;
#endif
        }

        bool flush()
        {
#if VKT_HAVE_VIRVO
            needFlush = false;
            vvFileIO fileio;
            return fileio.saveVolumeData(vd, true/*overwrite existing*/) == vvFileIO::OK;
#endif
            return false;
        }

        void resetVD()
        {
#if VKT_HAVE_VIRVO
            vd->setChan(1);
            if (vd->getFrameBytes() > 0)
            {
                auto vox = vd->vox;
                auto bpc = vd->bpc;
                std::string fn = vd->getFilename();
                uint8_t* data = new uint8_t[vd->getFrameBytes()];

                delete vd;
                vd = new vvVolDesc(fn.c_str(), vox.x, vox.y, vox.z,
                                   1, bpc, 1, &data,
                                   vvVolDesc::ARRAY_DELETE);
            }
#endif
        }

        bool needFlush = false;
        std::size_t readOffset = 0;
        std::size_t writeOffset = 0;
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
        if (!impl_->initialized() && !impl_->load())
            return 0;

        uint8_t* raw = impl_->vd->getRaw();

        // TODO: error / bounds checking
        std::memcpy(buf, raw + impl_->readOffset, len);

        impl_->readOffset += len;

        return len;
#else
        return 0;
#endif
    }

    std::size_t VirvoFile::write(char const* buf, std::size_t len)
    {
#if VKT_HAVE_VIRVO
        if (!impl_->initialized())
            return 0;

        uint8_t* raw = impl_->vd->getRaw();

        // TODO: error / bounds checking
        std::memcpy(raw + impl_->writeOffset, buf, len);

        impl_->writeOffset += len;

        impl_->needFlush = true;

        return len;
#else
        return 0;
#endif
    }

    bool VirvoFile::seek(std::size_t pos)
    {
#if VKT_HAVE_VIRVO
        impl_->readOffset = pos;
        impl_->writeOffset = pos;
        return true;
#else
        return false;
#endif
    }

    bool VirvoFile::flush()
    {
#if VKT_HAVE_VIRVO
        return impl_->flush();
#else
        return false;
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

    void VirvoFile::setDims(Vec3i dims)
    {
#if VKT_HAVE_VIRVO
        impl_->vd->vox[0] = dims.x;
        impl_->vd->vox[1] = dims.y;
        impl_->vd->vox[2] = dims.z;
        impl_->resetVD();
#endif
    }

    Vec3i VirvoFile::getDims()
    {
#if VKT_HAVE_VIRVO
        if (!impl_->initialized() && !impl_->load())
            return { 0, 0, 0 };

        return {
            (int)impl_->vd->vox[0],
            (int)impl_->vd->vox[1],
            (int)impl_->vd->vox[2]
            };
#else
        return {};
#endif
    }

    void VirvoFile::setDataFormat(DataFormat dataFormat)
    {
#if VKT_HAVE_VIRVO
        impl_->vd->bpc = getSizeInBytes(dataFormat);
        impl_->resetVD();
#endif
    }

    DataFormat VirvoFile::getDataFormat()
    {
#if VKT_HAVE_VIRVO
        if (!impl_->initialized() && !impl_->load())
            return DataFormat::Unspecified;

        if (impl_->vd->bpc == 1)
            return DataFormat::UInt8;
        else if (impl_->vd->bpc == 2)
            return DataFormat::UInt16;
        else if (impl_->vd->bpc == 4)
            return DataFormat::Float32;
        else
            return DataFormat::Unspecified;
#else
        return DataFormat::Unspecified;
#endif
    }

} // vkt
