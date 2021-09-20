// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "linalg.hpp"

namespace vkt
{
    namespace serial
    {
        template <typename Func>
        void for_each(int32_t xmin, int32_t xmax, Func func)
        {
            for (int32_t x = xmin; x != xmax; ++x)
            {
                func(x);
            }
        }

        template <typename Func>
        void for_each(int32_t xmin, int32_t xmax,
                      int32_t ymin, int32_t ymax,
                      int32_t zmin, int32_t zmax,
                      Func func)
        {
            for (int32_t z = zmin; z != zmax; ++z)
            {
                for (int32_t y = ymin; y != ymax; ++y)
                {
                    for (int32_t x = xmin; x != xmax; ++x)
                    {
                        func(x, y, z);
                    }
                }
            }
        }
    }

#ifdef __CUDACC__
    namespace cuda
    {
        template <typename Func>
        __global__ void for_each_kernel(int32_t xmin, int32_t xmax, Func func)
        {
            int32_t x = blockIdx.x * blockDim.x + threadIdx.x;

            if (x < xmin || x >= xmax)
                return;

            func(x);
        }

        template <typename Func>
        __global__ void for_each_kernel(int32_t xmin, int32_t xmax,
                                        int32_t ymin, int32_t ymax,
                                        int32_t zmin, int32_t zmax,
                                        Func func)
        {
            int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            int32_t z = blockIdx.z * blockDim.z + threadIdx.z;

            if (x < xmin || x >= xmax || y < ymin || y >= ymax || z < zmin || z >= zmax)
                return;

            func(x, y, z);
        }

        template <typename Func>
        void for_each(int32_t xmin, int32_t xmax, Func func)
        {
            dim3 blockSize = 256;
            dim3 gridSize = vkt::div_up(xmax-xmin, (int)blockSize.x);

            for_each_kernel<<<gridSize, blockSize>>>(xmin, xmax, func);
        }

        template <typename Func>
        void for_each(int32_t xmin, int32_t xmax,
                      int32_t ymin, int32_t ymax,
                      int32_t zmin, int32_t zmax,
                      Func func)
        {
            dim3 blockSize(8, 8, 8);
            dim3 gridSize(
                    vkt::div_up(xmax-xmin, (int)blockSize.x),
                    vkt::div_up(ymax-ymin, (int)blockSize.y),
                    vkt::div_up(zmax-zmin, (int)blockSize.z)
                    );

            for_each_kernel<<<gridSize, blockSize>>>(xmin, xmax,
                                                     ymin, ymax,
                                                     zmin, zmax,
                                                     func);
        }
    } // cuda
#endif

} // vkt


