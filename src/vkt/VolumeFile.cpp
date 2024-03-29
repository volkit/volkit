// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <string>

#include <boost/filesystem.hpp>

#include <vkt/VolumeFile.h>

#include <vkt/NiftiFile.hpp>
#include <vkt/RawFile.hpp>
#include <vkt/VirvoFile.hpp>
#include <vkt/VolumeFile.hpp>

#include "VolumeFile_impl.hpp"

enum FileType { RAW, Nifti, Virvo, Unknown };

static FileType getFileType(char const* fileName)
{
    boost::filesystem::path p(fileName);

    // raw

    static const std::string raw_extensions[] = { ".raw", ".RAW" };

    if (std::find(raw_extensions, raw_extensions + 2, p.extension()) != raw_extensions + 2)
        return RAW;

#if VKT_HAVE_NIFTI

    static const std::string nifti_extensions1[] = { ".nii", ".NII" };
    static const std::string nifti_extensions2[] = { ".gz", ".GZ" };

    if (std::find(nifti_extensions1, nifti_extensions1 + 2, p.extension()) != nifti_extensions1 + 2)
        boost::filesystem::change_extension(p, "");

    // Either extension removed, or not gzipped
    if (std::find(nifti_extensions2, nifti_extensions2 + 2, p.extension()) != nifti_extensions2 + 2)
        return Nifti;

#endif

#if VKT_HAVE_VIRVO

    static const std::string virvo_extensions[] = { ".avf", ".AVF", ".rvf", ".RVF", ".xvf", ".XVF" };

    if (std::find(virvo_extensions, virvo_extensions + 6, p.extension()) != virvo_extensions + 6)
        return Virvo;

#endif

    return Unknown;
}

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    VolumeFile::VolumeFile(char const* fileName, OpenMode om)
        : fileName_(fileName)
    {
        FileType ft = getFileType(fileName_);

        if (ft == RAW)
        {
            dataSource_ = new RawFile(fileName_, "rb");
            if (RawFile const* rf = dynamic_cast<RawFile const*>(dataSource_))
            {
                header_.isStructured = true;
                if (om == OpenMode::Read || om == OpenMode::ReadWrite)
                {
                    header_.dims = rf->getDims();
                    header_.dataFormat = rf->getDataFormat();
                }
            }
        }

#if VKT_HAVE_NIFTI
        if (ft == Nifti)
        {
            dataSource_ = new NiftiFile(fileName_);
            if (NiftiFile* nf = dynamic_cast<NiftiFile*>(dataSource_))
            {
                header_.isStructured = true;
                if (om == OpenMode::Read || om == OpenMode::ReadWrite)
                {
                    header_.dims = nf->getDims();
                    header_.dataFormat = nf->getDataFormat();
                }
            }
        }
#endif

#if VKT_HAVE_VIRVO
        // Also try to load Unknown file formats - Virvo might be able
        // to load them anyway!
        if (ft == Virvo || ft == Unknown)
        {
            dataSource_ = new VirvoFile(fileName_);
            if (VirvoFile* vf = dynamic_cast<VirvoFile*>(dataSource_))
            {
                header_.isStructured = true;
                if (om == OpenMode::Read || om == OpenMode::ReadWrite)
                {
                    header_.dims = vf->getDims();
                    header_.dataFormat = vf->getDataFormat();
                }
            }
        }
#endif
    }

    VolumeFile::~VolumeFile()
    {
        delete dataSource_;
    }

    std::size_t VolumeFile::read(char* buf, std::size_t len)
    {
        if (dataSource_ != nullptr)
            return dataSource_->read(buf, len);
        else
            return 0;
    }

    std::size_t VolumeFile::write(char const* buf, std::size_t len)
    {
        if (dataSource_ != nullptr)
            return dataSource_->write(buf, len);
        else
            return 0;
    }

    bool VolumeFile::seek(std::size_t pos)
    {
        if (dataSource_ != nullptr)
            return dataSource_->seek(pos);

        return false;
    }

    bool VolumeFile::flush()
    {
        if (dataSource_ != nullptr)
            return dataSource_->flush();

        return false;
    }

    bool VolumeFile::good() const
    {
        return dataSource_ != nullptr;
    }

    void VolumeFile::setHeader(VolumeFileHeader header)
    {
        FileType ft = getFileType(fileName_);

        if (ft == RAW)
        {
            if (RawFile* rf = dynamic_cast<RawFile*>(dataSource_))
            {
                rf->setDims(header_.dims);
                rf->setDataFormat(header_.dataFormat);
            }
        }

#if VKT_HAVE_NIFTI
        if (ft == Nifti)
        {
            if (NiftiFile* nf = dynamic_cast<NiftiFile*>(dataSource_))
            {
                nf->setDims(header_.dims);
                nf->setDataFormat(header_.dataFormat);
                nf->setVoxelMapping(header_.voxelMapping);
            }
        }
#endif

#if VKT_HAVE_VIRVO
        if (ft == Virvo || ft == Unknown)
        {
            if (VirvoFile* vf = dynamic_cast<VirvoFile*>(dataSource_))
            {
                vf->setDims(header_.dims);
                vf->setDataFormat(header_.dataFormat);
            }
        }
#endif
    }

    VolumeFileHeader VolumeFile::getHeader() const
    {
        return header_;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

void vktVolumeFileCreateS(vktVolumeFile* file, char const* fileName, vktOpenMode om)
{
    assert(file != nullptr);

    *file = new vktVolumeFile_impl(fileName, om);
}

vktDataSource vktVolumeFileGetBase(vktVolumeFile file)
{
    return file->base;
}

void vktVolumeFileDestroy(vktVolumeFile file)
{
    delete file;
}

size_t vktVolumeFileRead(vktVolumeFile file, char* buf, size_t len)
{
    return file->base->source->read(buf, len);
}

vktBool_t vktVolumeFileGood(vktVolumeFile file)
{
    return file->base->source->good() ? VKT_TRUE : VKT_FALSE;
}

vktVolumeFileHeader_t vktVolumeFileGetHeader(vktVolumeFile file)
{
    vktVolumeFileHeader_t chdr;
    vktVolumeFileHeaderDefaultInit(&chdr);

    vkt::VolumeFile const* vf = dynamic_cast<vkt::VolumeFile const*>(file->base->source);

    if (vf == nullptr)
        return chdr;

    vkt::VolumeFileHeader hdr = vf->getHeader();

    if (hdr.isStructured)
        chdr.isStructured = VKT_TRUE;
    else
        chdr.isStructured = VKT_FALSE;

    if (hdr.isHierarchical)
        chdr.isHierarchical = VKT_TRUE;
    else
        chdr.isHierarchical = VKT_FALSE;

    // Structured parameters
    vkt::Vec3i dims = hdr.dims;
    chdr.dims = { dims.x, dims.y, dims.z };
    chdr.dataFormat = (vktDataFormat)hdr.dataFormat;

    return chdr;
}
