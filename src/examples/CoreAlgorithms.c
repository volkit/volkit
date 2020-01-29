#include <string.h>

#include <vkt/Copy.h>
#include <vkt/Fill.h>
#include <vkt/ExecutionPolicy.h>
#include <vkt/Render.h>
#include <vkt/StructuredVolume.h>

#include "common.h"

int main()
{
    //--- Declarations ------------------------------------
    vktExecutionPolicy_t ep;
    vktStructuredVolume volume1;
    vktStructuredVolume volume2;

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
    VKT_SAFE_CALL(vktSetThreadExecutionPolicy(ep));*/

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

    // Render volume2
    vktRenderState_t renderState;
    memset(&renderState, 0, sizeof(renderState));
    VKT_SAFE_CALL(vktRenderSV(volume2, renderState, NULL));

    //--- Destroy volumes ---------------------------------

    VKT_SAFE_CALL(vktStructuredVolumeDestroy(volume1));
    VKT_SAFE_CALL(vktStructuredVolumeDestroy(volume2));
}
