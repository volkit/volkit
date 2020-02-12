#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common.hpp"
#include "linalg.hpp"

namespace vkt
{
    class VirvoFile : public DataSource
    {
    public:
        VirvoFile(char const* fileName);
       ~VirvoFile();

        virtual std::size_t read(char* buf, std::size_t len);
        virtual bool good() const;

        Vec3i getDims();

        uint16_t getBytesPerVoxel();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

    };

} // vkt
