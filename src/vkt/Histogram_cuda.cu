// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "HistogramView.hpp"
#include "Histogram_cuda.hpp"
#include "StructuredVolumeView.hpp"

namespace vkt
{
    __global__ void ComputeHistogram_kernel(
            StructuredVolumeView volume,
            HistogramView histogram,
            Vec3i first,
            Vec3i last
            )
    {
        int nx = last.x - first.x;
        int ny = last.y - first.y;
        int nz = last.z - first.z;

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x < nx && y < ny && z < nz)
        {
            float lo = volume.getVoxelMapping().x;
            float hi = volume.getVoxelMapping().y;

            std::size_t numBins = histogram.getNumBins();

            std::size_t* bins = histogram.getBinCounts();

            float val = volume.getValue(x, y, z);

            std::size_t binID = (std::size_t)((val - lo) * (numBins / (hi - lo)));

            atomicAdd((unsigned long long*)&bins[binID], 1ULL);
        }
    }

    void ComputeHistogramRange_cuda(
            StructuredVolume& volume,
            Histogram& histogram,
            Vec3i first,
            Vec3i last
            )
    {
        // Fill histogram with zeros
        std::size_t numBins = histogram.getNumBins();

        std::size_t* bins = histogram.getBinCounts();

        std::size_t zero = 0;

        MemsetRange(bins, &zero, sizeof(std::size_t) * numBins, sizeof(zero));

        // Compute histogram
        unsigned nx = last.x - first.x;
        unsigned ny = last.y - first.y;
        unsigned nz = last.z - first.z;

        dim3 blockSize(8, 8, 8);
        dim3 gridSize(
                div_up(nx, blockSize.x),
                div_up(ny, blockSize.y),
                div_up(nz, blockSize.z)
                );

        ComputeHistogram_kernel<<<gridSize, blockSize>>>(
                StructuredVolumeView(volume),
                HistogramView(histogram),
                first,
                last
                );
    }
}
