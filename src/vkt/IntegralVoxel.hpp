// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace vkt
{
    template <typename UI>
    class IntegralVoxel
    {
        static_assert(std::is_integral<UI>::value, "Type mismatch");
        static_assert(std::is_unsigned<UI>::value, "Type mismatch");

    public:
        IntegralVoxel(uint8_t const* vx)
        {
            std::memcpy(&uintRep_, vx, sizeof(UI));
        }

        template <typename UI2>
        UI2 as() const
        {
            static_assert(std::is_integral<UI2>::value, "Type mismatch");
            static_assert(std::is_unsigned<UI2>::value, "Type mismatch");

            return static_cast<UI2>(uintRep_);
        }

        /* implicit */ operator UI()
        {
            return uintRep_;
        }

    private:
        UI uintRep_;

    };

    template <typename UI>
    inline IntegralVoxel<UI> operator+(IntegralVoxel<UI> const& a, IntegralVoxel<UI> const& b)
    {
        UI tmp = a.template as<UI>() + b.template as<UI>();
        return IntegralVoxel<UI>((uint8_t const*)&tmp);
    }

    template <typename UI>
    inline IntegralVoxel<UI> operator-(IntegralVoxel<UI> const& a, IntegralVoxel<UI> const& b)
    {
        UI tmp = a.template as<UI>() - b.template as<UI>();
        return IntegralVoxel<UI>((uint8_t const*)&tmp);
    }

} // vkt
