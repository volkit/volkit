#include <stddef.h>

#include <vkt/Arithmetic.h>
#include <vkt/ExecutionPolicy.h>
#include <vkt/Render.h>
#include <vkt/StructuredVolume.h>
#include <vkt/Transform.h>

#define MAKE_CHECKERED_INIT(Level)                                                          \
    static void MakeCheckered##Level(int32_t x, int32_t y, int32_t z, vktVoxelView_t voxel) \
    {                                                                                       \
        x >>= Level;                                                                        \
        y >>= Level;                                                                        \
        z >>= Level;                                                                        \
                                                                                            \
        size_t linearIndex = z * 32 * 32 + y * 32 + x;                                      \
        if ((y % 2 == z % 2 && linearIndex % 2 == 0)                                        \
         || (y % 2 != z % 2 && linearIndex % 2 == 1))                                       \
            voxel.bytes[0] = 128;                                                           \
        else                                                                                \
            voxel.bytes[0] = 0;                                                             \
    }

MAKE_CHECKERED_INIT(2)
MAKE_CHECKERED_INIT(3)

int main()
{
    vktVec3i_t dims;
    int bpv;
    vktStructuredVolume volume1;
    vktStructuredVolume volume2;
    vktStructuredVolume volume3;
    vktStructuredVolume volume4;
    vktRenderState_t renderState;
    vktExecutionPolicy_t ep;

    dims.x = 32;
    dims.y = 32;
    dims.z = 32;
    bpv = 1;

    vktStructuredVolumeCreate(&volume1, dims.x, dims.y, dims.z, bpv,
                              1.f, 1.f, 1.f, 0.f, 1.f);
    vktTransformSV1(volume1, MakeCheckered3);

    vktStructuredVolumeCreate(&volume2, dims.x, dims.y, dims.z, bpv,
                              1.f, 1.f, 1.f, 0.f, 1.f);
    vktTransformSV1(volume2, MakeCheckered2);

    vktStructuredVolumeCreate(&volume3, dims.x, dims.y, dims.z, bpv,
                              1.f, 1.f, 1.f, 0.f, 1.f);
    // Compute sum on the CPU
    vktSumSV(volume3, volume1, volume2);

    vktStructuredVolumeCreate(&volume4, dims.x, dims.y, dims.z, bpv,
                              1.f, 1.f, 1.f, 0.f, 1.f);

    // Change execution policy in main thread to GPU
    ep = vktGetThreadExecutionPolicy();
    ep.device = vktExecutionPolicyDeviceGPU;
    //vktSetThreadExecutionPolicy(ep);
    // Compute sum on the GPU
    vktSafeSumSV(volume4, volume1, volume2);

    // Change execution policy in main thread back to CPU
    ep.device = vktExecutionPolicyDeviceCPU;
    //vktSetThreadExecutionPolicy(ep);

    // CPU rendering
    vktRenderStateDefaultInit(&renderState);
    renderState.renderAlgo = vktRenderAlgoMultiScattering;
    vktRenderSV(volume1, renderState, NULL);
    vktRenderSV(volume2, renderState, NULL);
    vktRenderSV(volume3, renderState, NULL);
    vktRenderSV(volume4, renderState, NULL);
}
