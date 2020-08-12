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

    struct Vec4i
    {
        int x;
        int y;
        int z;
        int w;
    };

    struct Box2f
    {
        Vec2f min;
        Vec2f max;
    };

    struct Box3f
    {
        Vec3f min;
        Vec3f max;
    };

    struct Box2i
    {
        Vec2i min;
        Vec2i max;
    };

    struct Box3i
    {
        Vec3i min;
        Vec3i max;
    };

    struct Mat3f
    {
        Vec3f col0;
        Vec3f col1;
        Vec3f col2;
    };

    struct Mat4f
    {
        Vec4f col0;
        Vec4f col1;
        Vec4f col2;
        Vec4f col3;
    };

    enum class Axis
    {
        X,
        Y,
        Z,
    };

} // vkt
