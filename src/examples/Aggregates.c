// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <stdio.h>
#include <stdlib.h>

#include <vkt/Aggregates.h>
#include <vkt/Decompose.h>
#include <vkt/InputStream.h>
#include <vkt/RawFile.h>
#include <vkt/StructuredVolume.h>


void printStatistics(vktAggregates_t aggr, int firstX,int firstY,int firstZ,
                                           int lastX,int lastY,int lastZ)
{
    printf("Range: (%d,%d,%d) -- (%d,%d,%d)\n",firstX,firstY,firstZ,lastX,lastY,lastZ);
    printf("Min. value: .......... %f\n", aggr.min);
    printf("Max. value: .......... %f\n", aggr.max);
    printf("Mean value: .......... %f\n", aggr.mean);
    printf("Standard deviation.... %f\n", aggr.stddev);
    printf("Variance: ............ %f\n", aggr.var);
    printf("Total sum: ........... %f\n", aggr.sum);
    printf("Total product: ....... %f\n", aggr.prod);
    printf("Min. value index: .... (%d,%d,%d)\n", aggr.argmin.x, aggr.argmin.y, aggr.argmin.z);
    printf("Max. value index: .... (%d,%d,%d)\n", aggr.argmax.x, aggr.argmax.y, aggr.argmax.z);
}

int main(int argc, char** argv)
{
    vktRawFile file;
    vktVec3i_t dims;
    uint16_t bpv;
    vktStructuredVolume volume;
    vktInputStream is;
    vktVec3i_t brickSize;
    vktVec3i_t decompDims;
    int x, y, z;
    int firstX, firstY, firstZ;
    int lastX, lastY, lastZ;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s file.raw\n", argv[0]);
        return EXIT_FAILURE;
    }

    vktRawFileCreateS(&file, argv[1], "r");

    dims = vktRawFileGetDims3iv(file);
    if (dims.x * dims.y * dims.z < 1)
    {
        fprintf(stderr, "%s", "Cannot parse dimensions from file name\n");
        return EXIT_FAILURE;
    }

    bpv = vktRawFileGetBytesPerVoxel(file);
    if (bpv == 0)
    {
        fprintf(stderr, "%s", "Cannot parse bytes per voxel from file name, guessing 1...\n");
        bpv = 1;
    }

    vktStructuredVolumeCreate(&volume, dims.x, dims.y, dims.z, bpv, 1.f, 1.f, 1.f, 0.f, 1.f);
    vktInputStreamCreate(&is, vktRawFileGetBase(file));
    vktInputStreamReadSV(is, volume);

    // Print statistics for the whole volume
    vktAggregates_t aggr;
    vktComputeAggregatesSV(volume, &aggr);
    printStatistics(aggr,0,0,0,dims.x,dims.y,dims.z);
    printf("\n");

    // Compute a brick decomposition and print per-brick statistics
    brickSize.x = 100;
    brickSize.y = 100;
    brickSize.z = 100;

    vktArray3D_vktStructuredVolume decomp;
    vktArray3D_vktStructuredVolume_CreateEmpty(&decomp);
    vktBrickDecomposeResizeSV(decomp, volume, brickSize.x,brickSize.y,brickSize.z,0,0,0,0,0,0);
    vktBrickDecomposeSV(decomp, volume, brickSize.x,brickSize.y,brickSize.z,0,0,0,0,0,0);

    decompDims = vktArray3D_vktStructuredVolume_Dims(decomp);
    for (z = 0; z < decompDims.z; ++z)
    {
        for (y = 0; y < decompDims.y; ++y)
        {
            for (x = 0; x < decompDims.x; ++x)
            {
                firstX = x * brickSize.x;
                firstY = y * brickSize.y;
                firstZ = z * brickSize.z;
                lastX = dims.x < firstX+brickSize.x ? dims.x : firstX+brickSize.x;
                lastY = dims.y < firstY+brickSize.y ? dims.y : firstY+brickSize.y;
                lastZ = dims.z < firstZ+brickSize.z ? dims.z : firstZ+brickSize.z;
                // Compute aggregates only for the brick range
                vktComputeAggregatesRangeSV(volume, &aggr,firstX,firstY,firstZ,
                                                          lastX,lastY,lastZ);
                printStatistics(aggr,firstX,firstY,firstZ,lastX,lastY,lastZ);
                printf("\n");
            }
        }
    }

    vktArray3D_vktStructuredVolume_Destroy(decomp);
    vktInputStreamDestroy(is);
    vktStructuredVolumeDestroy(volume);
    vktRawFileDestroy(file);
}
