// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cmath>
#include <ostream>

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

        float& operator[](int index)
        {
            return ((float*)(this))[index];
        }

        float const& operator[](int index) const
        {
            return ((float*)(this))[index];
        }

        float x;
        float y;
        float z;
    };

    inline Vec3f operator+(Vec3f const& a, Vec3f const& b)
    {
        return Vec3f(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline Vec3f operator-(Vec3f const& a, Vec3f const& b)
    {
        return Vec3f(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    inline Vec3f operator*(float a, Vec3f const& b)
    {
        return Vec3f(a * b.x, a * b.y, a * b.z);
    }

    inline Vec3f operator/(Vec3f const& a, float b)
    {
        return Vec3f(a.x / b, a.y / b, a.z / b);
    }

    inline float dot(Vec3f const& a, Vec3f const& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline float length(Vec3f const& a)
    {
        return sqrtf(dot(a, a));
    }

    inline Vec3f normalize(Vec3f const& a)
    {
        return a / length(a);
    }

    inline std::ostream& operator<<(std::ostream& out, Vec3f const& v)
    {
        out << '(' << v.x << ',' << v.y << ',' << v.z << ')';
        return out;
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


    struct Mat3f
    {
        Mat3f() {}

        Vec3f& operator()(int col)
        {
            return *((Vec3f*)(this) + col);
        }

        Vec3f const& operator()(int col) const
        {
            return *((Vec3f*)(this) + col);
        }

        float& operator()(int row, int col)
        {
            return (operator()(col))[row];
        }

        float const& operator()(int row, int col) const
        {
            return (operator()(col))[row];
        }

        Vec3f col0;
        Vec3f col1;
        Vec3f col2;
    };

    inline Vec3f operator*(Mat3f const& a, Vec3f const& b)
    {
        return Vec3f(
            a(0,0) * b.x + a(0,1) * b.y + a(0,2) * b.z,
            a(1,0) * b.x + a(1,1) * b.y + a(1,2) * b.z,
            a(2,0) * b.x + a(2,1) * b.y + a(2,2) * b.z
            );
    }

    inline std::ostream& operator<<(std::ostream& out, Mat3f const& m)
    {
        out << '(' << m.col0 << ',' << m.col1 << ',' << m.col2 << ')';
        return out;
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
