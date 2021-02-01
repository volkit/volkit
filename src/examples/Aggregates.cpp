// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <ostream>

#include <vkt/Aggregates.hpp>
#include <vkt/Decompose.hpp>
#include <vkt/InputStream.hpp>
#include <vkt/RawFile.hpp>
#include <vkt/StructuredVolume.hpp>


void printStatistics(vkt::Aggregates aggr,int firstX,int firstY,int firstZ,
                                          int lastX,int lastY,int lastZ)
{
    std::cout << "Range: (" << firstX << ',' << firstY << ',' << firstZ << ')'
                 << " -- (" << lastX << ',' << lastY << ',' << lastZ << ")\n";
    std::cout << "Min. value: .......... " << aggr.min << '\n';
    std::cout << "Max. value: .......... " << aggr.max << '\n';
    std::cout << "Mean value: .......... " << aggr.mean << '\n';
    std::cout << "Standard deviation: .. " << aggr.stddev << '\n';
    std::cout << "Variance: ............ " << aggr.var << '\n';
    std::cout << "Total sum: ........... " << aggr.sum << '\n';
    std::cout << "Total product: ....... " << aggr.prod << '\n';
    std::cout << "Min. value index: .... " << '(' << aggr.argmin.x << ','
                                                  << aggr.argmin.y << ','
                                                  << aggr.argmin.z << ")\n";
    std::cout << "Max. value index: .... " << '(' << aggr.argmax.x << ','
                                                  << aggr.argmax.y << ','
                                                  << aggr.argmax.z << ")\n";
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " file.raw\n";
        return EXIT_FAILURE;
    }

    vkt::RawFile file(argv[1], "r");

    vkt::Vec3i dims = file.getDims();
    if (dims.x * dims.y * dims.z < 1)
    {
        std::cerr << "Cannot parse dimensions from file name\n";
        return EXIT_FAILURE;
    }

    vkt::DataFormat dataFormat = file.getDataFormat();
    if (dataFormat == vkt::DataFormat::UInt8)
    {
        std::cerr << "Cannot parse data format from file name, guessing uint8...\n";
        dataFormat = vkt::DataFormat::UInt8;
    }

    vkt::StructuredVolume volume(dims.x, dims.y, dims.z, dataFormat);
    vkt::InputStream is(file);
    is.read(volume);

    // Print statistics for the whole volume
    vkt::Aggregates aggr; 
    vkt::ComputeAggregates(volume, aggr);
    printStatistics(aggr,0,0,0,dims.x,dims.y,dims.z);
    std::cout << '\n';

    // Compute a brick decomposition and print per-brick statistics
    vkt::Vec3i brickSize = { 100, 100, 100 };

    vkt::Array3D<vkt::StructuredVolume> decomp;
    vkt::BrickDecomposeResize(decomp, volume, brickSize);
    vkt::BrickDecompose(decomp, volume, brickSize);

    for (int z = 0; z < decomp.dims().z; ++z)
    {
        for (int y = 0; y < decomp.dims().y; ++y)
        {
            for (int x = 0; x < decomp.dims().x; ++x)
            {
                int firstX = x * brickSize.x;
                int firstY = y * brickSize.y;
                int firstZ = z * brickSize.z;
                int lastX = std::min(volume.getDims().x,firstX+brickSize.x);
                int lastY = std::min(volume.getDims().y,firstY+brickSize.y);
                int lastZ = std::min(volume.getDims().z,firstZ+brickSize.z);
                vkt::Aggregates aggr;
                // Compute aggregates only for the brick range
                vkt::ComputeAggregatesRange(volume, aggr,firstX,firstY,firstZ,
                                                         lastX,lastY,lastZ);
                printStatistics(aggr,firstX,firstY,firstZ,lastX,lastY,lastZ);
                std::cout << '\n';
            }
        }
    }
}
