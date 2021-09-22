// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/Rotate.hpp>
#include <vkt/StructuredVolume.hpp>

#include "for_each.hpp"
#include "linalg.hpp"
#include "StructuredVolumeView.hpp"

namespace vkt
{
    using serial::for_each;

    inline Mat3f quaternionToRotationMatrix(float re, Vec3f im)
    {
        float xx = im.x * im.x;
        float xy = im.x * im.y;
        float xz = im.x * im.z;
        float xw = im.x * re;
        float yy = im.y * im.y;
        float yz = im.y * im.z;
        float yw = im.y * re;
        float zz = im.z * im.z;
        float zw = im.z * re;
        float ww =   re * re;

        Mat3f result;

        result.col0.x = 2.f * (ww + xx) - 1.f;
        result.col0.y = 2.f * (xy + zw);
        result.col0.z = 2.f * (xz - yw);
        result.col1.x = 2.f * (xy - zw);
        result.col1.y = 2.f * (ww + yy) - 1.f;
        result.col1.z = 2.f * (yz + xw);
        result.col2.x = 2.f * (xz + yw);
        result.col2.y = 2.f * (yz - xw);
        result.col2.z = 2.f * (ww + zz) - 1.f;

        return result;
    }

    void RotateRange_serial(
            StructuredVolume& dest,
            StructuredVolume& source,
            Vec3i first,
            Vec3i last,
            Vec3f axis,
            float angleInRadians,
            Vec3f centerOfRotation
            )
    {
        StructuredVolumeView destView(dest);
        // So we can use sampleLinear()
        StructuredVolumeView sourceView(source);

        // Normalize axis
        axis = normalize(axis);

        // Inverse angle as we'll later deal with the inverse transform
        angleInRadians = 6.28319f/*360deg*/ - angleInRadians;

        // To quaternion
        float quatRe = cosf(angleInRadians * .5f);
        Vec3f quatIm{
            axis.x * sinf(angleInRadians * .5f),
            axis.y * sinf(angleInRadians * .5f),
            axis.z * sinf(angleInRadians * .5f)
            };

        // Compute rotation matrix from axis and angle
        Mat3f rot = quaternionToRotationMatrix(quatRe, quatIm);

        // Iterate over the _whole_ dest volume,
        // apply the _inverse_ rotation, and reconstruct
        // if the rotated position is inside [first..last)
        for_each(0,dest.getDims().x,0,dest.getDims().y,0,dest.getDims().z,
                 [=] (int x, int y, int z) mutable {
                     Vec3f p{ (float)x, (float)y, (float)z };
                     p = p - centerOfRotation;
                     p = rot * p;
                     p = p + centerOfRotation;

                     if (p.x >= first.x && p.x < last.x
                      && p.y >= first.y && p.y < last.y
                      && p.z >= first.z && p.z < last.z)
                     {
                         float val = sourceView.sampleLinear(p.x, p.y, p.z);
                         destView.setValue(x, y, z, val);
                     }
                });
    }
} // vkt
