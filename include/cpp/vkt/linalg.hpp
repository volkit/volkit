// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

namespace vkt
{
    struct Vec2f
    {
        float x;
        float y;
    };

    struct Vec3f
    {
        float x;
        float y;
        float z;
    };

    struct Vec4f
    {
        float x;
        float y;
        float z;
        float w;
    };

    struct Vec2i
    {
        int x;
        int y;
    };

    struct Vec3i
    {
        int x;
        int y;
        int z;
    };

    struct Box3f
    {
        Vec3f min;
        Vec3f max;
    };

    struct Mat3f
    {
        Vec3f col0;
        Vec3f col1;
        Vec3f col2;
    };

    enum class Axis
    {
        X,
        Y,
        Z,
    };

} // vkt
