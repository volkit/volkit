#include <cstddef>

#include <vkt/Arithmetic.hpp>
#include <vkt/ExecutionPolicy.hpp>
#include <vkt/Render.hpp>
#include <vkt/StructuredVolume.hpp>
#include <vkt/Transform.hpp>

template <unsigned Level>
static void MakeCheckered(int32_t x, int32_t y, int32_t z, vkt::VoxelView voxel)
{
    x >>= Level;
    y >>= Level;
    z >>= Level;

    std::size_t linearIndex = z * 32 * 32 + y * 32 + x;
    if ((y % 2 == z % 2 && linearIndex % 2 == 0)
     || (y % 2 != z % 2 && linearIndex % 2 == 1))
        voxel.bytes[0] = 128;
    else
        voxel.bytes[0] = 0;
}

int main()
{
    vkt::Vec3i dims = { 32, 32, 32 };

    vkt::DataFormat dataFormat = vkt::DataFormat::UInt8;
    vkt::StructuredVolume volume1(dims.x, dims.y, dims.z, dataFormat);
    vkt::Transform(volume1, MakeCheckered<3>);

    vkt::StructuredVolume volume2(dims.x, dims.y, dims.z, dataFormat);
    vkt::Transform(volume2, MakeCheckered<2>);

    vkt::StructuredVolume volume3(dims.x, dims.y, dims.z, dataFormat);
    // Compute sum on the CPU
    vkt::Sum(volume3, volume1, volume2);

    vkt::StructuredVolume volume4(dims.x, dims.y, dims.z, dataFormat);

    // Change execution policy in main thread to GPU
    vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
    ep.device = vkt::ExecutionPolicy::Device::GPU;
    //vkt::SetThreadExecutionPolicy(ep);
    // Compute sum on the GPU
    vkt::SafeSum(volume4, volume1, volume2);

    // Change execution policy in main thread back to CPU
    ep.device = vkt::ExecutionPolicy::Device::CPU;
    //vkt::SetThreadExecutionPolicy(ep);

    // CPU rendering
    vkt::RenderState renderState;
    renderState.renderAlgo = vkt::RenderAlgo::MultiScattering;
    vkt::Render(volume1, renderState);
    vkt::Render(volume2, renderState);
    vkt::Render(volume3, renderState);
    vkt::Render(volume4, renderState);
}
