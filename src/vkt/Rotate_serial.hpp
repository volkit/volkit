// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/Rotate.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
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

        result(0, 0) = 2.f * (ww + xx) - 1.f;
        result(1, 0) = 2.f * (xy + zw);
        result(2, 0) = 2.f * (xz - yw);
        result(0, 1) = 2.f * (xy - zw);
        result(1, 1) = 2.f * (ww + yy) - 1.f;
        result(2, 1) = 2.f * (yz + xw);
        result(0, 2) = 2.f * (xz + yw);
        result(1, 2) = 2.f * (yz - xw);
        result(2, 2) = 2.f * (ww + zz) - 1.f;

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
        // Normalize axis
        axis = normalize(axis);

        // Inverse angle as we'll later deal with the inverse transform
        angleInRadians = 6.28319f/*360deg*/ - angleInRadians;

        // To quaternion
        float quatRe = cosf(angleInRadians * .5f);
        Vec3f quatIm(
            axis.x * sinf(angleInRadians * .5f),
            axis.y * sinf(angleInRadians * .5f),
            axis.z * sinf(angleInRadians * .5f)
            );

        // Compute rotation matrix from axis and angle
        Mat3f rot = quaternionToRotationMatrix(quatRe, quatIm);

        // Iterate over the _whole_ dest volume,
        // apply the _inverse_ rotation, and reconstruct
        // if the rotated position is inside [first..last)
        for (int z = 0; z < dest.getDims().z; ++z)
        {
            for (int y = 0; y < dest.getDims().y; ++y)
            {
                for (int x = 0; x < dest.getDims().x; ++x)
                {
                    Vec3f p(x, y, z);
                    p = p - centerOfRotation;
                    p = rot * p;
                    p = p + centerOfRotation;

                    if (p.x >= first.x && p.x < last.x
                     && p.y >= first.y && p.y < last.y
                     && p.z >= first.z && p.z < last.z)
                    {
                        float val = source.sampleLinear(p.x, p.y, p.z);
                        dest.setValue(x, y, z, val);
                    }
                }
            }
        }
    }
} // vkt
