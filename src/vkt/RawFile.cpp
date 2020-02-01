#include <vkt/RawFile.hpp>

#include <vkt/RawFile.h>

namespace vkt
{
    RawFile::RawFile(char const* fileName, char const* mode)
        : fileName_(fileName)
        , mode_(mode)
    {
        file_ = fopen(fileName_, mode_);
    }

    RawFile::RawFile(FILE* file)
        : file_(file)
    {
        file_ = fopen(fileName_, mode_);
    }

    RawFile::~RawFile()
    {
        fclose(file_);
    }

    std::size_t RawFile::read(char* buf, std::size_t len)
    {
        return fread(buf, len, 1, file_);
    }

    bool RawFile::good() const
    {
        return file_ != nullptr;
    }

} // vkt
