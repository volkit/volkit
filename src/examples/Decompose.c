#include <vkt/Decompose.h>
#include <vkt/Fill.h>
#include <vkt/linalg.h>
#include <vkt/Render.h>
#include <vkt/StructuredVolume.h>

int main()
{
    // Volume dimensions
    vktVec3i_t dims = { .x=120,.y=66,.z=49 };

    // Brick size
    vktVec3i_t brickSize = { .x=16, .y=16, .z=16 };

    // Halo / ghost cells
    vktVec3i_t haloSizeNeg = { .x=1, .y=1, .z=1 };
    vktVec3i_t haloSizePos = { .x=1, .y=1, .z=1 };

    int bpv = 1;

    float mappingLo = 0.f;
    float mappingHi = 1.f;

    float distX = 1.f;
    float distY = 1.f;
    float distZ = 1.f;

    vktStructuredVolume volume;

    // The destination data structure
    vktArray3D_vktStructuredVolume decomp;

    vktStructuredVolumeCreate(
            &volume,
            dims.x,
            dims.y,
            dims.z,
            bpv,
            distX,
            distY,
            distZ,
            mappingLo,
            mappingHi
            );

    vktArray3D_vktStructuredVolume_CreateEmpty(&decomp);

    // Put some values in
    vktFillSV(volume, .1f);

    // Preallocate storage for the decomposition
    vktBrickDecomposeResizeSV(
            decomp,
            volume,
            brickSize.x,
            brickSize.y,
            brickSize.z,
            haloSizeNeg.x,
            haloSizeNeg.y,
            haloSizeNeg.z,
            haloSizePos.x,
            haloSizePos.y,
            haloSizePos.z
            );

    // Compute the decomposition
    vktBrickDecomposeSV(
            decomp,
            volume,
            brickSize.x,
            brickSize.y,
            brickSize.z,
            haloSizeNeg.x,
            haloSizeNeg.y,
            haloSizeNeg.z,
            haloSizePos.x,
            haloSizePos.y,
            haloSizePos.z
            );

    vktVec3i_t index = { .x=0, .y=0, .z=0 };
    vktRenderState_t renderState;
    vktRenderStateDefaultInit(&renderState);
    vktRenderSV(
        *vktArray3D_vktStructuredVolume_Access(decomp, index),
        renderState,
        NULL
        );

    vktArray3D_vktStructuredVolume_Destroy(decomp);
    vktStructuredVolumeDestroy(volume);
}
