#include "math.h"

#include <vkt/Fill.h>
#include <vkt/LookupTable.h>
#include <vkt/Render.h>
#include <vkt/Rotate.h>
#include <vkt/StructuredVolume.h>

int main(void)
{
    // Create a structured volume
    vktVec3i_t dims = { .x=256,.y=128,.z=100 };

    int bpv = 1;
    vktStructuredVolume volume, rotatedVolume;

    vktVec3f_t axis = { .x=1.f,.y=.3f,.z=0.f };
    vktVec3f_t centerOfRotation = { .x=dims.x*.5f,.y=dims.y*.5f,.z=dims.z*.5f };

    float rgba[20];
    vktLookupTable lut;
    vktRenderState_t renderState;

    vktStructuredVolumeCreate(
            &volume,
            dims.x, dims.y, dims.z,
            bpv,
            1.f, 1.f, 1.f, // dist
            0.f, 1.f // mapping
            );

    vktFillSV(volume, .1f);

    vktFillRangeSV(
            volume,
            64,4,4,
            192,124,96,
            1.f
            );

    // Destination volume; has the same size as the original one
    vktStructuredVolumeCreate(
            &rotatedVolume,
            dims.x, dims.y, dims.z,
            bpv,
            1.f, 1.f, 1.f,
            0.f, 1.f
            );

    vktFillSV(rotatedVolume, 1.f);

    // Rotate the volume with rotation center in the middle
    vktRotateSV(
            rotatedVolume,
            volume,
            axis,                   // rotation axis
            45.f * M_PI / 180.f,    // rotation angle in radians
            centerOfRotation        // center of rotation
            );

    rgba[ 0] =  1.f; rgba[ 1] = 1.f; rgba[ 2] = 1.f;  rgba[ 3] = .005f;
    rgba[ 4] =  0.f; rgba[ 5] = .1f; rgba[ 6] = .1f;  rgba[ 7] = .25f;
    rgba[ 8] =  .5f; rgba[ 9] = .5f; rgba[10] = .7f;  rgba[11] = .5f;
    rgba[12] =  .7f; rgba[13] = .7f; rgba[14] = .07f; rgba[15] = .75f;
    rgba[16] =  1.f; rgba[17] = .3f; rgba[18] = .3f;  rgba[19] = 1.f;

    vktLookupTableCreate(&lut,5,1,1,vktColorFormatRGBA32F);
    vktLookupTableSetData(lut,(uint8_t*)rgba);

    vktRenderStateDefaultInit(&renderState);
    renderState.renderAlgo = vktRenderAlgoMultiScattering;
    renderState.rgbaLookupTable = vktLookupTableGetResourceHandle(lut);
    vktRenderSV(rotatedVolume, renderState, NULL);

    vktLookupTableDestroy(lut);
    vktStructuredVolumeDestroy(rotatedVolume);
    vktStructuredVolumeDestroy(volume);
}
