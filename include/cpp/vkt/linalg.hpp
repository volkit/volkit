#pragma once

namespace vkt
{
    struct vec2f
    {
        vec2f() {}
        vec2f(float x, float y)
            : x(x)
            , y(y)
        {
        }

        float x;
        float y;
    };

    inline bool operator==(vec2f const& a, vec2f const& b)
    {
        return a.x == b.x && a.y == b.y;
    }


    struct vec3i
    {
        vec3i() {}
        vec3i(int a)
            : x(a)
            , y(a)
            , z(a)
        {
        }

        vec3i(int x, int y, int z)
            : x(x)
            , y(y)
            , z(z)
        {
        }

        int x;
        int y;
        int z;
    };

    inline bool operator==(vec3i const& a, vec3i const& b)
    {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    inline vec3i operator+(vec3i const& a, vec3i const& b)
    {
        return vec3i(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline vec3i operator-(vec3i const& a, vec3i const& b)
    {
        return vec3i(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    inline vec3i operator*(vec3i const& a, vec3i const& b)
    {
        return vec3i(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    inline vec3i& operator+=(vec3i& a, vec3i const& b)
    {
        a = a + b;
        return a;
    }

    inline vec3i& operator-=(vec3i& a, vec3i const& b)
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

} // vkt
