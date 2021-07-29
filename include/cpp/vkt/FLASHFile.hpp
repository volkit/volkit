// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <memory>

#include <vkt/HierarchicalVolume.hpp> // for Brick

#include "common.hpp"
#include "linalg.hpp"

namespace vkt
{
    class FLASHFile : public DataSource
    {
    public:
        FLASHFile(char const* fileName, char const* var);
       ~FLASHFile();

        virtual std::size_t read(char* buf, std::size_t len);
        virtual std::size_t write(char const* buf, std::size_t len);
        virtual bool seek(std::size_t pos);
        virtual bool flush();
        virtual bool good() const;

        // void setNumSubBricks(unsigned numSubBricks);

        unsigned getNumBricks() const;

        // Data is copied!
        // void setBricks(Brick const* bricks);

        Brick const* getBricks() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

    };

} // vkt
