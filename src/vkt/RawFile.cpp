#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include <vkt/RawFile.hpp>

#include <vkt/RawFile.h>

static std::vector<std::string> string_split(std::string s, char delim)
{
    std::vector<std::string> result;

    std::istringstream stream(s);

    for (std::string token; std::getline(stream, token, delim); )
    {
        result.push_back(token);
    }

    return result;
}

namespace vkt
{
    RawFile::RawFile(char const* fileName, char const* mode)
        : fileName_(fileName)
        , mode_(mode)
    {
        file_ = fopen(fileName_, mode_);

        // Try to parse dimensions and bpv from file name
        std::vector<std::string> strings;

        strings = string_split(fileName_, '_');

        for (auto str : strings)
        {
            int32_t dimx = 0;
            int32_t dimy = 0;
            int32_t dimz = 0;
            uint16_t bpv = 0;
            int res = 0;

            // Dimensions
            res = sscanf(str.c_str(), "%dx%dx%d", &dimx, &dimy, &dimz);
            if (res == 3)
                dims_ = Vec3i(dimx, dimy, dimz);

            res = sscanf(str.c_str(), "int%hu", &bpv);
            if (res == 1)
                bytesPerVoxel_ = bpv / 8;

            res = sscanf(str.c_str(), "uint%hu", &bpv);
            if (res == 1)
                bytesPerVoxel_ = bpv / 8;
        }
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

    Vec3i RawFile::getDims() const
    {
        return dims_;
    }

    uint16_t RawFile::getBytesPerVoxel() const
    {
        return bytesPerVoxel_;
    }

} // vkt
