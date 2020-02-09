// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdint>

#include "Arithmetic_cuda.hpp"
#include "StructuredVolumeView.hpp"

namespace vkt
{
    template <typename Func>
    __global__ void ArithmeticOp_kernel(
            StructuredVolumeView dest,
            StructuredVolumeView source1,
            StructuredVolumeView source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset,
            Func func
            )
    {
        int nx = last.x - first.x;
        int ny = last.y - first.y;
        int nz = last.z - first.z;

        int x = (blockIdx.x * blockDim.x + threadIdx.x) - first.x;
        int y = (blockIdx.y * blockDim.y + threadIdx.y) - first.y;
        int z = (blockIdx.z * blockDim.z + threadIdx.z) - first.z;

        if (x < nx && y < ny && z < nz)
        {
            float val1;
            float val2;

            source1.getValue(x, y, z, val1);
            source2.getValue(x, y, z, val2);

            float val3 = func(val1, val2);

            dest.setValue(
                x + dstOffset.x,
                y + dstOffset.y,
                z + dstOffset.z,
                val3
                );
        }
    }

    template <typename Func>
    void ArithmeticOp(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset,
            Func func
            )
    {
        unsigned nx = last.x - first.x;
        unsigned ny = last.y - first.y;
        unsigned nz = last.z - first.z;

        dim3 blockSize(8, 8, 8);
        dim3 gridSize(
                div_up(nx, blockSize.x),
                div_up(ny, blockSize.y),
                div_up(nz, blockSize.z)
                );

        ArithmeticOp_kernel<<<gridSize, blockSize>>>(
                StructuredVolumeView(dest),
                StructuredVolumeView(source1),
                StructuredVolumeView(source2),
                first,
                last,
                dstOffset,
                func
                );
    }

    void SumRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        ArithmeticOp(
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset,
            [] __device__ (float f1, float f2) { return f1 + f2; }
            );
    }

    void DiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        ArithmeticOp(
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset,
            [] __device__ (float f1, float f2) { return f1 - f2; }
            );
    }

    void ProdRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        ArithmeticOp(
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset,
            [] __device__ (float f1, float f2) { return f1 * f2; }
            );
    }

    void QuotRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        ArithmeticOp(
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset,
            [] __device__ (float f1, float f2) { return f1 / f2; }
            );
    }

    void AbsDiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        ArithmeticOp(
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset,
            [] __device__ (float f1, float f2) { return fabsf(f1 - f2); }
            );
    }

    void SafeSumRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        float lo = dest.getVoxelMapping().x;
        float hi = dest.getVoxelMapping().y;

        ArithmeticOp(
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset,
            [lo, hi] __device__ (float f1, float f2) { return clamp(f1 + f2, lo, hi); }
            );
    }

    void SafeDiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        float lo = dest.getVoxelMapping().x;
        float hi = dest.getVoxelMapping().y;

        ArithmeticOp(
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset,
            [lo, hi] __device__ (float f1, float f2) { return clamp(f1 - f2, lo, hi); }
            );
    }

    void SafeProdRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        float lo = dest.getVoxelMapping().x;
        float hi = dest.getVoxelMapping().y;

        ArithmeticOp(
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset,
            [lo, hi] __device__ (float f1, float f2) { return clamp(f1 * f2, lo, hi); }
            );
    }

    void SafeQuotRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        float lo = dest.getVoxelMapping().x;
        float hi = dest.getVoxelMapping().y;

        ArithmeticOp(
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset,
            [lo, hi] __device__ (float f1, float f2) { return clamp(f1 / f2, lo, hi); }
            );
    }

    void SafeAbsDiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        float lo = dest.getVoxelMapping().x;
        float hi = dest.getVoxelMapping().y;

        ArithmeticOp(
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset,
            [lo, hi] __device__ (float f1, float f2) { return clamp(fabsf(f1 - f2), lo, hi); }
            );
    }

} // vkt
