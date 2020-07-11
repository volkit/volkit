// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cmath>
#include <ostream>

#include <vkt/linalg.h>

#include "macros.hpp"

namespace vkt
{
    //--- General -----------------------------------------

    template <typename T>
    VKT_FUNC inline T min(T const& a, T const& b)
    {
        return b < a ? b : a;
    }

    template <typename T>
    VKT_FUNC inline T max(T const& a, T const& b)
    {
        return a < b ? b : a;
    }

    template <typename T, typename S>
    VKT_FUNC inline T lerp(T const& a, T const& b, S const& x)
    {
        return (S(1.0f) - x) * a + x * b;
    }

    template <typename T>
    VKT_FUNC inline T clamp(T const& x, T const& a, T const& b)
    {
        return max(a, min(x, b));
    }

    template <typename T>
    VKT_FUNC inline T saturate(T const& x)
    {
        return clamp(x, T(0.0), T(1.0));
    }

    template <typename T>
    VKT_FUNC inline T div_up(T a, T b)
    {
        return (a + b - 1) / b;
    }


    //--- Vec2f -------------------------------------------

    VKT_FUNC inline bool operator==(Vec2f const& a, Vec2f const& b)
    {
        return a.x == b.x && a.y == b.y;
    }


    //--- Vec3f -------------------------------------------

    VKT_FUNC inline Vec3f operator+(Vec3f const& a, Vec3f const& b)
    {
        return  { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    VKT_FUNC inline Vec3f operator-(Vec3f const& a, Vec3f const& b)
    {
        return  { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    VKT_FUNC inline Vec3f operator*(float a, Vec3f const& b)
    {
        return  { a * b.x, a * b.y, a * b.z };
    }

    VKT_FUNC inline Vec3f operator/(Vec3f const& a, float b)
    {
        return { a.x / b, a.y / b, a.z / b };
    }

    VKT_FUNC inline float dot(Vec3f const& a, Vec3f const& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    VKT_FUNC inline float length(Vec3f const& a)
    {
        return sqrtf(dot(a, a));
    }

    VKT_FUNC inline Vec3f normalize(Vec3f const& a)
    {
        return a / length(a);
    }

    inline std::ostream& operator<<(std::ostream& out, Vec3f const& v)
    {
        out << '(' << v.x << ',' << v.y << ',' << v.z << ')';
        return out;
    }


    //--- Vec3i -------------------------------------------

    VKT_FUNC inline bool operator==(Vec3i const& a, Vec3i const& b)
    {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    VKT_FUNC inline Vec3i operator+(Vec3i const& a, Vec3i const& b)
    {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    VKT_FUNC inline Vec3i operator-(Vec3i const& a, Vec3i const& b)
    {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    VKT_FUNC inline Vec3i operator*(Vec3i const& a, Vec3i const& b)
    {
        return { a.x * b.x, a.y * b.y, a.z * b.z };
    }

    VKT_FUNC inline Vec3i& operator+=(Vec3i& a, Vec3i const& b)
    {
        a = a + b;
        return a;
    }

    VKT_FUNC inline Vec3i& operator-=(Vec3i& a, Vec3i const& b)
    {
        a = a - b;
        return a;
    }

    inline std::ostream& operator<<(std::ostream& out, Vec3i const& v)
    {
        out << '(' << v.x << ',' << v.y << ',' << v.z << ')';
        return out;
    }


    //--- vktVec3i_t --------------------------------------

    VKT_FUNC inline vktVec3i_t operator+(vktVec3i_t const& a, vktVec3i_t const& b)
    {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    VKT_FUNC inline vktVec3i_t operator-(vktVec3i_t const& a, vktVec3i_t const& b)
    {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    VKT_FUNC inline vktVec3i_t operator*(vktVec3i_t const& a, vktVec3i_t const& b)
    {
        return { a.x * b.x, a.y * b.y, a.z * b.z };
    }

    VKT_FUNC inline vktVec3i_t& operator+=(vktVec3i_t& a, vktVec3i_t const& b)
    {
        a = a + b;
        return a;
    }

    VKT_FUNC inline vktVec3i_t& operator-=(vktVec3i_t& a, vktVec3i_t const& b)
    {
        a = a - b;
        return a;
    }


    //--- Vec3f -------------------------------------------

    VKT_FUNC inline Vec3f operator*(Mat3f const& a, Vec3f const& b)
    {
        return { 
            a.col0.x * b.x + a.col1.x * b.y + a.col2.x * b.z,
            a.col0.y * b.x + a.col1.y * b.y + a.col2.y * b.z,
            a.col0.z * b.x + a.col1.y * b.y + a.col2.z * b.z
            };
    }

    inline std::ostream& operator<<(std::ostream& out, Mat3f const& m)
    {
        out << '(' << m.col0 << ',' << m.col1 << ',' << m.col2 << ')';
        return out;
    }

} // vkt
