#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vkt/ExecutionPolicy.h>
#include <vkt/InputStream.h>
#include <vkt/LookupTable.h>
#include <vkt/Render.h>
#include <vkt/StructuredVolume.h>
#include <vkt/VolumeFile.h>

int main(int argc, char** argv)
{
    vktVolumeFile file;
    vktVolumeFileHeader_t hdr;
    vktVec3i_t dims;
    uint16_t bpv;
    vktStructuredVolume volume;
    vktInputStream is;
    float rgba[20];
    vktLookupTable lut;
    vktRenderState_t renderState;
    vktExecutionPolicy_t ep;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s file.raw\n", argv[0]);
        return EXIT_FAILURE;
    }

    vktVolumeFileCreateS(&file, argv[1], vktOpenModeRead);

    hdr = vktVolumeFileGetHeader(file);

    if (!hdr.isStructured)
    {
        fprintf(stderr, "%s", "No valid volume file\n");
        return EXIT_FAILURE;
    }

    dims = hdr.dims;
    if (dims.x * dims.y * dims.z < 1)
    {
        fprintf(stderr, "%s", "Cannot parse dimensions from file name\n");
        return EXIT_FAILURE;
    }

    bpv = hdr.bytesPerVoxel;
    if (bpv == 0)
    {
        fprintf(stderr, "%s", "Cannot parse bytes per voxel from file name, guessing 1...\n");
        bpv = 1;
    }

    vktStructuredVolumeCreate(&volume, dims.x, dims.y, dims.z, bpv, 1.f, 1.f, 1.f, 0.f, 1.f);
    vktInputStreamCreate(&is, vktVolumeFileGetBase(file));
    vktInputStreamReadSV(is, volume);

    rgba[ 0] =  1.f; rgba[ 1] = 1.f; rgba[ 2] = 1.f;  rgba[ 3] = .005f;
    rgba[ 4] =  0.f; rgba[ 5] = .1f; rgba[ 6] = .1f;  rgba[ 7] = .25f;
    rgba[ 8] =  .5f; rgba[ 9] = .5f; rgba[10] = .7f;  rgba[11] = .5f;
    rgba[12] =  .7f; rgba[13] = .7f; rgba[14] = .07f; rgba[15] = .75f;
    rgba[16] =  1.f; rgba[17] = .3f; rgba[18] = .3f;  rgba[19] = 1.f;

    vktLookupTableCreate(&lut,5,1,1,vktColorFormatRGBA32F);
    vktLookupTableSetData(lut,(uint8_t*)rgba);

    // Switch execution to GPU (remove those lines for CPU rendering)
    memset(&ep, 0, sizeof(ep));
    ep.device = vktExecutionPolicyDeviceGPU;
    vktSetThreadExecutionPolicy(ep);

    vktRenderStateDefaultInit(&renderState);
    //renderState.renderAlgo = vktRenderAlgoRayMarching;
    //renderState.renderAlgo = vktRenderAlgoImplicitIso;
    renderState.renderAlgo = vktRenderAlgoMultiScattering;
    renderState.rgbaLookupTable = vktLookupTableGetResourceHandle(lut);
    vktRenderSV(volume, renderState, NULL);

    vktLookupTableDestroy(lut);
    vktInputStreamDestroy(is);
    vktStructuredVolumeDestroy(volume);
    vktVolumeFileDestroy(file);
}
