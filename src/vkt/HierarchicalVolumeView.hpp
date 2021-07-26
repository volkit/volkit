// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cassert>
#include <iostream>
#include <ostream>

#include <visionaray/math/forward.h>
#include <visionaray/math/io.h>
#include <visionaray/math/ray.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/bvh.h>

#include <vkt/HierarchicalVolume.hpp>

#include "ArrayView1D.hpp"
#include "StructuredVolumeView.hpp"

namespace visionaray
{
    struct Ray : basic_ray<float>
    {
        float* sumWeightedValues;
        float* sumWeights;
        vec3f* sumDerivatives;
        vec3f* sumDerivativeCoefficients;
    };

    struct ActiveBrickRegion
    {
        int primID;
        int level;
        aabb bounds; // w/o halo
        aabb domain; // w/ halo
        vkt::ArrayView1D<vkt::Brick> bricks;
        uint8_t* scalars; // pointer to the whole field!
    };

    inline aabb get_bounds(ActiveBrickRegion const& abr)
    {
        return abr.domain;
    }

    void split_primitive(aabb& /*L*/, aabb& /*R*/, float /*plane*/, int /*axis*/, ActiveBrickRegion const& /*box*/)
    {
        assert(0);
    }

    struct HitRecord : hit_record<Ray, primitive<unsigned>>
    {
    };

    inline void VKT_FUNC add(float &sumWeights,
                             float &sumWeightedValues,
                             float weight,
                             float scalar)
    {
        sumWeights += weight;
        sumWeightedValues += weight*scalar;
    }

    inline void VKT_FUNC add(vec3f &sumWeights,
                             vec3f &sumWeightedValues,
                             vec3f weight,
                             vec3f scalar)
    {
        sumWeights += weight;
        sumWeightedValues += weight*scalar;
    }

    inline VKT_FUNC float getScalar(const vkt::Brick& brick, const uint8_t* scalars,
                                    const int ix, const int iy, const int iz)
    {
        // Treat as floats
        std::size_t begin = brick.offsetInBytes / sizeof(float);

        const std::size_t idx
          = begin
          + ix
          + iy * brick.dims.x
          + iz * brick.dims.x * brick.dims.y;

        return ((float*)(scalars))[idx];
    }

    template<bool NEED_DERIVATIVE=true>
    inline VKT_FUNC void addBasisFunctions(float &sumWeightedValues,
                                           float &sumWeights,
                                           vec3f &sumDerivatives,
                                           vec3f &sumDerivativeCoefficients,
                                           const vkt::Brick& brick,
                                           ActiveBrickRegion const& abr,
                                           const vec3f pos)
    {
        vec3i brickLower = {brick.lower.x,brick.lower.y,brick.lower.z};
        vec3i brickSize = {brick.dims.x,brick.dims.y,brick.dims.z};

        const float cellWidth = (1<<abr.level);
        //const float invCellWidth = 1.f/cellWidth;
        const vec3f localPos = (pos - vec3f(brickLower)) / vec3f(cellWidth) - vec3f(0.5f);
#if 0
        const vec3i idx_hi   = vec3i(localPos+vec3f(1.f)); // +1 to emulate 'floor()'
        const vec3i idx_lo   = idx_hi - vec3i(1);
#else
        vec3i idx_lo   = vec3i(floorf(localPos.x),floorf(localPos.y),floorf(localPos.z));
        idx_lo = max(vec3i(-1), idx_lo);
        const vec3i idx_hi   = idx_lo + vec3i(1);
#endif
        const vec3f frac     = localPos - vec3f(idx_lo);
        const vec3f neg_frac = vec3f(1.f) - frac;

        // #define INV_CELL_WIDTH invCellWidth
#define INV_CELL_WIDTH 1.f
        if (idx_lo.z >= 0 && idx_lo.z < brickSize.z) {
            if (idx_lo.y >= 0 && idx_lo.y < brickSize.y) {
                if (idx_lo.x >= 0 && idx_lo.x < brickSize.x) {
                    const float scalar = getScalar(brick,abr.scalars,idx_lo.x,idx_lo.y,idx_lo.z);
                    if (true) {
                        const float weight = (neg_frac.z)*(neg_frac.y)*(neg_frac.x);
                        // sumWeights += weight;
                        // sumWeightedValues += weight * getScalar(brickID,idx_lo.x,idx_lo.y,idx_lo.z);
                        // if (NEED_DERIVATIVE) {
                        //     const float dx = (neg_frac.z)*(neg_frac.y)*(-INV_CELL_WIDTH);
                        //     const float dy = (neg_frac.z)*(neg_frac.x)*(-INV_CELL_WIDTH);
                        //     const float dz = (neg_frac.y)*(neg_frac.x)*(-INV_CELL_WIDTH);
                        //     add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
                        // }
                        add(sumWeights,sumWeightedValues,weight,scalar);
                    }
                }
                if (idx_hi.x < brickSize.x) {
                    const float scalar = getScalar(brick,abr.scalars,idx_hi.x,idx_lo.y,idx_lo.z);
                    if (true) {
                        const float weight = (neg_frac.z)*(neg_frac.y)*(frac.x);
                        // sumWeights += weight;
                        // sumWeightedValues += weight * getScalar(brickID,idx_hi.x,idx_lo.y,idx_lo.z);
                        // if (NEED_DERIVATIVE) {
                        //     const float dx = (neg_frac.z)*(neg_frac.y)*(+INV_CELL_WIDTH);
                        //     const float dy = (neg_frac.z)*(    frac.x)*(-INV_CELL_WIDTH);
                        //     const float dz = (neg_frac.y)*(    frac.x)*(-INV_CELL_WIDTH);
                        //     add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
                        // }
                        add(sumWeights,sumWeightedValues,weight,scalar);
                    }
                }
            }
            if (idx_hi.y < brickSize.y) {
                if (idx_lo.x >= 0 && idx_lo.x < brickSize.x) {
                    const float scalar = getScalar(brick,abr.scalars,idx_lo.x,idx_hi.y,idx_lo.z);
                    if (true) {
                        const float weight = (neg_frac.z)*(frac.y)*(neg_frac.x);
                        // sumWeights += weight;
                        // sumWeightedValues += weight * getScalar(brickID,idx_lo.x,idx_hi.y,idx_lo.z);
                        // if (NEED_DERIVATIVE) {
                        //     const float dx = (neg_frac.z)*(    frac.y)*(-INV_CELL_WIDTH);
                        //     const float dy = (neg_frac.z)*(neg_frac.x)*(+INV_CELL_WIDTH);
                        //     const float dz = (    frac.y)*(neg_frac.x)*(-INV_CELL_WIDTH);
                        //     add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
                        // }
                        add(sumWeights,sumWeightedValues,weight,scalar);
                    }
                }
                if (idx_hi.x < brickSize.x) {
                    const float scalar = getScalar(brick,abr.scalars,idx_hi.x,idx_hi.y,idx_lo.z);
                    if (true) {
                        const float weight = (neg_frac.z)*(frac.y)*(frac.x);
                        // sumWeights += weight;
                        // sumWeightedValues += weight * getScalar(brickID,idx_hi.x,idx_hi.y,idx_lo.z);
                        // if (NEED_DERIVATIVE) {
                        //     const float dx = (neg_frac.z)*(    frac.y)*(+INV_CELL_WIDTH);
                        //     const float dy = (neg_frac.z)*(    frac.x)*(+INV_CELL_WIDTH);
                        //     const float dz = (    frac.y)*(    frac.x)*(-INV_CELL_WIDTH);
                        //     add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
                        // }
                        add(sumWeights,sumWeightedValues,weight,scalar);
                    }
                }
            }
        }

        if (idx_hi.z < brickSize.z) {
            if (idx_lo.y >= 0 && idx_lo.y < brickSize.y) {
                if (idx_lo.x >= 0 && idx_lo.x < brickSize.x) {
                    const float scalar = getScalar(brick,abr.scalars,idx_lo.x,idx_lo.y,idx_hi.z);
                    if (true) {
                        const float weight = (frac.z)*(neg_frac.y)*(neg_frac.x);
                        // sumWeights += weight;
                        // sumWeightedValues += weight * getScalar(brickID,idx_lo.x,idx_lo.y,idx_hi.z);
                        // if (NEED_DERIVATIVE) {
                        //     const float dx = (    frac.z)*(neg_frac.y)*(-INV_CELL_WIDTH);
                        //     const float dy = (    frac.z)*(neg_frac.x)*(-INV_CELL_WIDTH);
                        //     const float dz = (neg_frac.y)*(neg_frac.x)*(+INV_CELL_WIDTH);
                        //     add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
                        // }
                        add(sumWeights,sumWeightedValues,weight,scalar);
                    }
                }
                if (idx_hi.x < brickSize.x) {
                    const float scalar = getScalar(brick,abr.scalars,idx_hi.x,idx_lo.y,idx_hi.z);
                    if (true) {
                        const float weight = (frac.z)*(neg_frac.y)*(frac.x);
                        // sumWeights += weight;
                        // sumWeightedValues += weight * getScalar(brickID,idx_hi.x,idx_lo.y,idx_hi.z);
                        // if (NEED_DERIVATIVE) {
                        //     const float dx = (    frac.z)*(neg_frac.y)*(+INV_CELL_WIDTH);
                        //     const float dy = (    frac.z)*(    frac.x)*(-INV_CELL_WIDTH);
                        //     const float dz = (neg_frac.y)*(    frac.x)*(+INV_CELL_WIDTH);
                        //     add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
                        // }
                        add(sumWeights,sumWeightedValues,weight,scalar);
                    }
                }
            }
            if (idx_hi.y < brickSize.y) {
                if (idx_lo.x >= 0 && idx_lo.x < brickSize.x) {
                    const float scalar = getScalar(brick,abr.scalars,idx_lo.x,idx_hi.y,idx_hi.z);
                    if (true) {
                        const float weight = (frac.z)*(frac.y)*(neg_frac.x);
                        // sumWeights += weight;
                        // sumWeightedValues += weight * getScalar(brickID,idx_lo.x,idx_hi.y,idx_hi.z);
                        // if (NEED_DERIVATIVE) {
                        //     const float dx = (    frac.z)*(    frac.y)*(-INV_CELL_WIDTH);
                        //     const float dy = (    frac.z)*(neg_frac.x)*(+INV_CELL_WIDTH);
                        //     const float dz = (    frac.y)*(neg_frac.x)*(+INV_CELL_WIDTH);
                        //     add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
                        // }
                        add(sumWeights,sumWeightedValues,weight,scalar);
                    }
                }
                if (idx_hi.x < brickSize.x) {
                    const float scalar = getScalar(brick,abr.scalars,idx_hi.x,idx_hi.y,idx_hi.z);
                    if (true) {
                        const float weight = (frac.z)*(frac.y)*(frac.x);
                        // sumWeights += weight;
                        // sumWeightedValues += weight * getScalar(brickID,idx_hi.x,idx_hi.y,idx_hi.z);
                        // if (NEED_DERIVATIVE) {
                        //     const float dx = (    frac.z)*(    frac.y)*(+INV_CELL_WIDTH);
                        //     const float dy = (    frac.z)*(    frac.x)*(+INV_CELL_WIDTH);
                        //     const float dz = (    frac.y)*(    frac.x)*(+INV_CELL_WIDTH);
                        //     add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
                        // }
                        add(sumWeights,sumWeightedValues,weight,scalar);
                    }
                }
            }
        }
    }

    HitRecord intersect(Ray const& ray, ActiveBrickRegion const& abr)
    {
        if (ray.ori.x >= abr.domain.min.x && ray.ori.x <= abr.domain.max.x
         && ray.ori.y >= abr.domain.min.y && ray.ori.y <= abr.domain.max.y
         && ray.ori.z >= abr.domain.min.z && ray.ori.z <= abr.domain.max.z)
        {
            addBasisFunctions<false>(*ray.sumWeightedValues,*ray.sumWeights,
                                     *ray.sumDerivatives,*ray.sumDerivativeCoefficients,
                                     abr.bricks[abr.primID],abr,ray.ori);
        }
        return {};
    }
}

namespace vkt
{
    class HierarchicalVolumeView
    {
    public:
        HierarchicalVolumeView() = default;
        VKT_FUNC HierarchicalVolumeView(HierarchicalVolume& volume)
            : data_(volume.getData())
        {
            bricks_ = ArrayView1D<Brick>(volume.getBricks(), volume.getNumBricks());

            using namespace visionaray;

            aligned_vector<ActiveBrickRegion> abrs(volume.getNumBricks());

            for (std::size_t i = 0; i < volume.getNumBricks(); ++i)
            {
                int primID(i);
                int level(volume.getBricks()[i].level);
                Vec3i lower = volume.getBricks()[i].lower;
                Vec3i dims = volume.getBricks()[i].dims;

                Box3f bounds = { {0.f,0.f,0.f}, {(float)dims.x,(float)dims.y,(float)dims.z} };
                bounds.min = bounds.min * (float)(1<<level);
                bounds.max = bounds.max * (float)(1<<level);
                bounds.min += {(float)lower.x,(float)lower.y,(float)lower.z};
                bounds.max += {(float)lower.x,(float)lower.y,(float)lower.z};

                Box3f domain = { {0.f,0.f,0.f}, {(float)dims.x,(float)dims.y,(float)dims.z} };
                domain.min -= Vec3f{.5f, .5f, .5f};
                domain.max += Vec3f{.5f, .5f, .5f};
                domain.min = domain.min * (float)(1<<level);
                domain.max = domain.max * (float)(1<<level);
                domain.min += {(float)lower.x,(float)lower.y,(float)lower.z};
                domain.max += {(float)lower.x,(float)lower.y,(float)lower.z};

                abrs[i] = { primID, level,
                            { {bounds.min.x,bounds.min.y,bounds.min.z},
                              {bounds.max.x,bounds.max.y,bounds.max.z} },
                            { {domain.min.x,domain.min.y,domain.min.z},
                              {domain.max.x,domain.max.y,domain.max.z} },
                            bricks_, data_ };
            }

            binned_sah_builder builder;
            cpuBVH = builder.build(visionaray::index_bvh<visionaray::ActiveBrickRegion>{},
                                   abrs.data(), abrs.size());
            cpuBVHRef = cpuBVH.ref();

#ifdef __CUDACC__
            gpuBVH = visionaray::cuda_index_bvh<visionaray::ActiveBrickRegion>(cpuBVH);
            gpuBVHRef = gpuBVH.ref();
#endif
        }

        VKT_FUNC float sampleLinear(int32_t x, int32_t y, int32_t z)
        {
            using namespace visionaray;

            float sumWeightedValues = 0.f;
            float sumWeights = 0.f;
            vec3f sumDerivatives(0.f);
            vec3f sumDerivativeCoefficients(0.f);

            Ray r;
            r.ori = vec3(x,y,z);
            r.dir = normalize(vec3(1.f));
            r.tmin = 0.f;
            r.tmax = 2e-10f;
            r.sumWeightedValues = &sumWeightedValues;
            r.sumWeights = &sumWeights;
            r.sumDerivatives = &sumDerivatives;
            r.sumDerivativeCoefficients = &sumDerivativeCoefficients;

            intersect(r, cpuBVHRef);

            return sumWeights >= 1e-20f ? sumWeightedValues/sumWeights : 0.f;
        }

    private:
        uint8_t* data_;

        ArrayView1D<Brick> bricks_;

        visionaray::index_bvh<visionaray::ActiveBrickRegion> cpuBVH;
        visionaray::index_bvh_ref_t<visionaray::ActiveBrickRegion> cpuBVHRef;

#ifdef __CUDACC__
        visionaray::cuda_index_bvh<visionaray::ActiveBrickRegion> gpuBVH;
        visionaray::cuda_index_bvh_ref_t<visionaray::ActiveBrickRegion> gpuBVHRef;
#endif
    };

} // vkt
