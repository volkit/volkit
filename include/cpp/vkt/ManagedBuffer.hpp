#pragma once

#include <cstddef>
#include <cstdint>

#include <vkt/ExecutionPolicy.hpp>

namespace vkt
{

    /*!
     * @brief  a buffer whose memory can be migrated between devices on demand
     */

    class ManagedBuffer
    {
    public:
        /*!
         * @brief  if device changed, copy data arrays into new address space
         */
        void migrate();

        void resize(std::size_t size);

        void copy(ManagedBuffer& rhs);

    protected:
        void allocate(std::size_t size);
        void deallocate();

        uint8_t* data_ = nullptr;
        std::size_t size_ = 0;

    private:
        ExecutionPolicy lastAllocationPolicy_ = {};

    };

} // vkt
