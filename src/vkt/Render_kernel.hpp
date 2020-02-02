// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <visionaray/math/aabb.h>
#include <visionaray/math/intersect.h>
#include <visionaray/math/limits.h>
#include <visionaray/math/vector.h>
#include <visionaray/phase_function.h>
#include <visionaray/result_record.h>

#include <vkt/Render.hpp>


//-------------------------------------------------------------------------------------------------
// Ray marching with absorption plus emission model
//

template <typename Volume>
struct RayMarchingKernel
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
    }

    visionaray::aabb bbox;
    Volume volume;
    //Transfunc transfunc;
    float dt;
};


//-------------------------------------------------------------------------------------------------
// Implicit iso-surface rendering
//

template <typename Volume>
struct ImplicitIsoKernel
{
    template <typename T>
    VSNRAY_FUNC
    inline visionaray::vector<3, T> gradient(visionaray::vector<3, T> tex_coord)
    {
        using namespace visionaray;

        vector<3, T> s1;
        vector<3, T> s2;

        float DELTA = 0.01f;

        s1.x = tex3D(volume, tex_coord + vector<3, T>(DELTA, 0.0f, 0.0f));
        s2.x = tex3D(volume, tex_coord - vector<3, T>(DELTA, 0.0f, 0.0f));
        s1.y = tex3D(volume, tex_coord + vector<3, T>(0.0f, DELTA, 0.0f));
        s2.y = tex3D(volume, tex_coord - vector<3, T>(0.0f, DELTA, 0.0f));
        s1.z = tex3D(volume, tex_coord + vector<3, T>(0.0f, 0.0f, DELTA));
        s2.z = tex3D(volume, tex_coord - vector<3, T>(0.0f, 0.0f, DELTA));

        return s2 - s1;
    }

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

        S last(-1e20f);

        S isoT(-1e20f);

        while (any(t < hit_rec.tfar))
        {
            // sample volume
            S voxel = convert_to_float(tex3D(volume, tex_coord));

            // normalize to [0..1]
            voxel /= S(numeric_limits<typename Volume::value_type>::max());

            if (any(last >= S(-1e10f)))
            {
                for (uint16_t i = 0; i < numIsoSurfaces; ++i)
                {
                    if ((last <= isoSurfaces[i] && voxel >= isoSurfaces[i])
                     || (last >= isoSurfaces[i] && voxel <= isoSurfaces[i]))
                    {
                        isoT = t;
                        vector<3, S> N = normalize(gradient(tex_coord));
                        vector<3, S> ka(S(.2f));
                        vector<3, S> kd(dot(N, -ray.dir) * voxel);
                        result.color = C(ka + kd, S(1.f));
                    }
                }
            }

            if (all(isoT >= S(-1e10f)))
                break;

            // step on
            tex_coord += inc;
            t += dt;
            last = voxel;
        }

        result.hit = isoT >= S(-1e10f);
        return result;
    }

    visionaray::aabb bbox;
    Volume volume;
    uint16_t numIsoSurfaces;
    float isoSurfaces[vkt::RenderState::MaxIsoSurfaces];
    float dt;
};


//-------------------------------------------------------------------------------------------------
// Simple multi-scattering
// Loosely based on M. Raab: Ray Tracing Inhomogeneous Volumes, RTGems I (2019)
//

template <typename Volume, typename Transfunc>
struct MultiScatteringKernel
{
    float mu(visionaray::vec3 const& pos)
    {
        using namespace visionaray;

        float voxel = convert_to_float(tex3D(volume, pos / bbox.size()));

        // normalize to [0..1]
        voxel /= float(numeric_limits<typename Volume::value_type>::max());

        return voxel;
    }

    template <typename Ray>
    bool sample_interaction(Ray& r, float d, visionaray::random_generator<float>& gen)
    {
        using namespace visionaray;

        float t = 0.0f;
        vec3 pos;

        do
        {
            t -= log(1.0f - gen.next()) / mu_;

            pos = r.ori + r.dir * t;
            if (t >= d)
            {
                return false;
            }
        }
        while (mu(pos) < gen.next() * mu_);

        r.ori = pos;
        return true;
    };

    template <typename Ray>
    auto operator()(Ray r, visionaray::random_generator<float>& gen, int x, int y)
    {
        using namespace visionaray;

        using S = typename Ray::scalar_type;
        using C = vector<4, S>;

        henyey_greenstein<float> f;
        f.g = 0.f; // isotropic

        result_record<S> result;

        vec3 throughput(1.f);

        auto hit_rec = intersect(r, bbox);

        if (any(hit_rec.hit))
        {
            r.ori += r.dir * hit_rec.tnear;
            hit_rec.tfar -= hit_rec.tnear;

            unsigned bounce = 0;

            while (sample_interaction(r, hit_rec.tfar, gen))
            {
                // Is the path length exceeded?
                if (bounce++ >= 1024)
                {
                    throughput = vec3(0.0f);
                    break;
                }

                vec3 alb(0.f);
                
                if (albedo)
                {
                    vec4 rgba = tex1D(albedo, mu(r.ori));
                    alb = rgba.xyz() * rgba.w;
                }
                else
                    alb = vec3(1.f-mu(r.ori));

                throughput *= alb;
                // Russian roulette absorption
                float prob = max_element(throughput);
                if (prob < 0.2f)
                {
                    if (gen.next() > prob)
                    {
                        throughput = vec3(0.0f);
                        break;
                    }
                    throughput /= prob;
                }

                // Sample phase function
                vec3 scatter_dir;
                float pdf;
                f.sample(-r.dir, scatter_dir, pdf, gen);
                r.dir = scatter_dir;

                hit_rec = intersect(r, bbox);
            }
        }

        // Look up the environment
        float t = y / heightf_;
        vec3 Ld = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
        vec3 L = Ld * throughput;

        result.color = vec4(L, 1.0f);
        result.hit = hit_rec.hit;
        return result;
    }

    visionaray::aabb bbox;
    Volume volume;
    Transfunc albedo;
    float mu_;
    float heightf_;
};
