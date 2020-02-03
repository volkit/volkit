// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

namespace vkt
{
    struct Vec2f
    {
        Vec2f() {}
        Vec2f(float x, float y)
            : x(x)
            , y(y)
        {
        }

        float x;
        float y;
    };

    inline bool operator==(Vec2f const& a, Vec2f const& b)
    {
        return a.x == b.x && a.y == b.y;
    }


    struct Vec3f
    {
        Vec3f() {}
        Vec3f(float x, float y, float z)
            : x(x)
            , y(y)
            , z(z)
        {
        }

        float x;
        float y;
        float z;
    };

    inline Vec3f operator+(Vec3f const& a, Vec3f const& b)
    {
        return Vec3f(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline Vec3f operator*(float a, Vec3f const& b)
    {
        return Vec3f(a * b.x, a * b.y, a * b.z);
    }


    struct Vec4f
    {
        Vec4f() {}
        Vec4f(float x, float y, float z, float w)
            : x(x)
            , y(y)
            , z(z)
            , w(w)
        {
        }

        float x;
        float y;
        float z;
        float w;
    };


    struct Vec2i
    {
        Vec2i() {}
        Vec2i(int a)
            : x(a)
            , y(a)
        {
        }

        Vec2i(int x, int y)
            : x(x)
            , y(y)
        {
        }

        int x;
        int y;
    };


    struct Vec3i
    {
        Vec3i() {}
        Vec3i(int a)
            : x(a)
            , y(a)
            , z(a)
        {
        }

        Vec3i(int x, int y, int z)
            : x(x)
            , y(y)
            , z(z)
        {
        }

        int x;
        int y;
        int z;
    };

    inline bool operator==(Vec3i const& a, Vec3i const& b)
    {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    inline Vec3i operator+(Vec3i const& a, Vec3i const& b)
    {
        return Vec3i(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline Vec3i operator-(Vec3i const& a, Vec3i const& b)
    {
        return Vec3i(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    inline Vec3i operator*(Vec3i const& a, Vec3i const& b)
    {
        return Vec3i(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    inline Vec3i& operator+=(Vec3i& a, Vec3i const& b)
    {
        a = a + b;
        return a;
    }

    inline Vec3i& operator-=(Vec3i& a, Vec3i const& b)
    {
        a = a - b;
        return a;
    }

    template <typename T>
    inline T min(T const& a, T const& b)
    {
        return b < a ? b : a;
    }

    template <typename T>
    inline T max(T const& a, T const& b)
    {
        return a < b ? b : a;
    }

    template <typename T, typename S>
    inline T lerp(T const& a, T const& b, S const& x)
    {
        return (S(1.0f) - x) * a + x * b;
    }

    template <typename T>
    inline T clamp(T const& x, T const& a, T const& b)
    {
        return max(a, min(x, b));
    }

    template <typename T>
    inline T div_up(T a, T b)
    {
        return (a + b - 1) / b;
    }

    enum class Axis
    {
        X,
        Y,
        Z,
    };

} // vkt
