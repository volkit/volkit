// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <visionaray/math/aabb.h>
#include <visionaray/math/intersect.h>
#include <visionaray/math/limits.h>
#include <visionaray/math/vector.h>

template <typename Volume>
struct RenderKernel
{
    template <typename Ray>
    auto operator()(Ray ray)
    {
        using namespace visionaray;

        using S = typename Ray::scalar_type;
        using C = vector<4, S>;

        result_record<S> result;

        auto hit_rec = intersect(ray, bbox);
        auto t = hit_rec.tnear;

        result.color = C(0.0);

        vector<3, S> boxSize(bbox.size());
        vector<3, S> pos = ray.ori + ray.dir * t;
        vector<3, S> tex_coord = pos / boxSize;

        vector<3, S> inc = ray.dir * S(dt) / boxSize;

        while (any(t < hit_rec.tfar))
        {
            // sample volume
            S voxel = convert_to_float(tex3D(volume, tex_coord));

            // normalize to [0..1]
            voxel /= S(numeric_limits<typename Volume::value_type>::max());

            // classification
            C color(voxel, voxel, voxel, voxel);

            // opacity correction
            color.w = S(1.f) - pow(S(1.f) - color.w, S(dt));

            // premultiplied alpha
            color.xyz() *= color.w;

            // front-to-back alpha compositing
            result.color += select(
                    t < hit_rec.tfar,
                    color * (1.0f - result.color.w),
                    C(0.0)
                    );

            // early-ray termination - don't traverse w/o a contribution
            if (all(result.color.w >= 0.999))
            {
                break;
            }

            // step on
            tex_coord += inc;
            t += dt;
        }

        result.hit = hit_rec.hit;
        return result;
    };

    visionaray::aabb bbox;
    Volume volume;
    //Transfunc transfunc;
    float dt;
};
