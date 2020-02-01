#include <string.h>

#include <vkt/Copy.h>
#include <vkt/Fill.h>
#include <vkt/Flip.h>
#include <vkt/ExecutionPolicy.h>
#include <vkt/Render.h>
#include <vkt/StructuredVolume.h>
#include <vkt/Transform.h>

#include "common.h"

static void TransformOp1(int32_t x, int32_t y, int32_t z, uint8_t* voxel)
{
    if (x == y && y == z)
        voxel[0] = 0xFF;
}

static void TransformOp2(int32_t x, int32_t y, int32_t z, uint8_t* voxel1, uint8_t* voxel2)
{
    voxel1[0] |= voxel2[0];
    voxel2[0]  = voxel1[0];
}

int main()
{
    //--- Declarations ------------------------------------
    vktExecutionPolicy_t ep;
    vktStructuredVolume volume1;
    vktStructuredVolume volume2;
    vktStructuredVolume volume3;
    vktRenderState_t renderState;

    // Bytes per voxel
    int bpv;

    // Mapping for highest/lowest voxel value
    float mappingLo;
    float mappingHi;


    //--- Create a volume ---------------------------------

    bpv = 1;
    mappingLo = 0.f;
    mappingHi = 1.f;

    VKT_SAFE_CALL(vktStructuredVolumeCreate(&volume1,
                                            64, 64, 64,
                                            bpv,
                                            mappingLo,
                                            mappingHi));

    //--- Fill --------------------------------------------

    // Set GPU execution policy
    /*memset(&ep, 0, sizeof(ep));
    ep.device = vktExecutionPolicyDeviceGPU;
    vktSetThreadExecutionPolicy(ep);*/

    // Fill the whole volume
    VKT_SAFE_CALL(vktFillSV(volume1, .1f));

    //--- CopyRange ---------------------------------------

    // Create a 2nd volume
    VKT_SAFE_CALL(vktStructuredVolumeCreate(&volume2,
                                            24, 24, 24,
                                            bpv,
                                            mappingLo,
                                            mappingHi));

    // Copy range from volume1 to volume2
    VKT_SAFE_CALL(vktCopyRangeSV(volume2,
                                 volume1,
                                 10, 10, 10,
                                 34, 34, 34, 
                                 0, 0, 0));

    //--- TransformRange ----------------------------------

    // Iterate over a range and apply a unary operation
    VKT_SAFE_CALL(vktTransformRangeSV1(volume2,
                                       2, 2, 2,
                                       22, 22, 22,
                                       TransformOp1));

    // Create a copy; copy construction can be used to
    // copy volumes if all parameters (bpv, dims, etc.)
    // of the two volumes match exactly
    VKT_SAFE_CALL(vktStructuredVolumeCreateCopy(&volume3,
                                                volume2));

    // Flip volume3 (not a core algorithm)
    VKT_SAFE_CALL(vktFlipSV(volume3, vktAxisX));

    // Transform w/ binary operation
    VKT_SAFE_CALL(vktTransformSV2(volume2,
                                  volume3,
                                  TransformOp2));

    //--- Render (not core) -------------------------------

    // Render volume2
    memset(&renderState, 0, sizeof(renderState));
    VKT_SAFE_CALL(vktRenderSV(volume3, renderState, NULL));

    //--- Destroy volumes ---------------------------------

    VKT_SAFE_CALL(vktStructuredVolumeDestroy(volume1));
    VKT_SAFE_CALL(vktStructuredVolumeDestroy(volume2));
    VKT_SAFE_CALL(vktStructuredVolumeDestroy(volume3));
}
