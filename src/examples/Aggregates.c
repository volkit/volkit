// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <stdio.h>
#include <stdlib.h>

#include <vkt/Aggregates.h>
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
    vktInputStreamCreateF(&is, file);
    vktInputStreamReadSV(is, volume);

    // Print statistics for the whole volume
    vktAggregates_t aggr;
    vktComputeAggregatesSV(volume, &aggr);
    printStatistics(aggr,0,0,0,dims.x,dims.y,dims.z);
    // TODO: brick decomposition (cf. C++ and Python examples)

    vktInputStreamDestroy(is);
    vktStructuredVolumeDestroy(volume);
    vktRawFileDestroy(file);
}
