#include <stdlib.h>

#include <vkt/Fill.h>
#include <vkt/Render.h>
#include <vkt/Scan.h>
#include <vkt/StructuredVolume.h>

#include "common.h"

int main(void)
{
    vktStructuredVolume volume;
    vktRenderState_t renderState;

    // Bytes per voxel
    int bpv;

    // Mapping for highest/lowest voxel value
    float mappingLo;
    float mappingHi;

    // Voxel distance
    float distX;
    float distY;
    float distZ;


    //--- Create a volume ---------------------------------

    bpv = 1;
    mappingLo = 0.f;
    mappingHi = 1.f;
    distX = 1.f;
    distY = 1.f;
    distZ = 1.f;

    VKT_SAFE_CALL(vktStructuredVolumeCreate(&volume,
                                            8, 8, 8,
                                            bpv,
                                            distX,
                                            distY,
                                            distZ,
                                            mappingLo,
                                            mappingHi));

    // Fill the volume
    VKT_SAFE_CALL(vktFillSV(volume, .02f));

    //--- ScanRange ---------------------------------------

    // Note how dst and src are the same
    VKT_SAFE_CALL(vktScanRangeSV(volume, // dst
                                 volume, // src
                                 0, 0, 0,
                                 4, 4, 4,
                                 0, 0, 0));

    // In the following, some components of first > last
    VKT_SAFE_CALL(vktScanRangeSV(volume,
                                 volume,
                                 7, 0, 0,
                                 3, 4, 4,
                                 0, 0, 0));

    VKT_SAFE_CALL(vktScanRangeSV(volume,
                                 volume,
                                 0, 7, 0,
                                 4, 3, 4,
                                 0, 0, 0));

    VKT_SAFE_CALL(vktScanRangeSV(volume,
                                 volume,
                                 0, 0, 7,
                                 4, 4, 3,
                                 0, 0, 0));

    VKT_SAFE_CALL(vktScanRangeSV(volume,
                                 volume,
                                 7, 7, 0,
                                 3, 3, 4,
                                 0, 0, 0));

    VKT_SAFE_CALL(vktScanRangeSV(volume,
                                 volume,
                                 0, 7, 7,
                                 4, 3, 3,
                                 0, 0, 0));

    VKT_SAFE_CALL(vktScanRangeSV(volume,
                                 volume,
                                 7, 0, 7,
                                 3, 4, 3,
                                 0, 0, 0));

    VKT_SAFE_CALL(vktScanRangeSV(volume,
                                 volume,
                                 7, 7, 7,
                                 3, 3, 3,
                                 0, 0, 0));

    //--- Render ------------------------------------------

    // Render volume
    vktRenderStateDefaultInit(&renderState);
    VKT_SAFE_CALL(vktRenderSV(volume, renderState, NULL));

    //--- Destroy volumes ---------------------------------

    VKT_SAFE_CALL(vktStructuredVolumeDestroy(volume));
}
