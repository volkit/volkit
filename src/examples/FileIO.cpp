#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>

#include <vkt/InputStream.hpp>
#include <vkt/LookupTable.hpp>
#include <vkt/RawFile.hpp>
#include <vkt/Render.hpp>
#include <vkt/StructuredVolume.hpp>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " file.raw\n";
        return EXIT_FAILURE;
    }

    vkt::RawFile file(argv[1], "r");

    vkt::Vec3i dims = file.getDims();
    if (dims.x * dims.y * dims.z < 1)
    {
        std::cerr << "Cannot parse dimensions from file name\n";
        return EXIT_FAILURE;
    }

    uint16_t bpv = file.getBytesPerVoxel();
    if (bpv == 0)
    {
        std::cerr << "Cannot parse bytes per voxel from file name, guessing 1...\n";
        bpv = 1;
    }

    vkt::StructuredVolume volume(dims.x, dims.y, dims.z, bpv);
    vkt::InputStream is(file);
    is.read(volume);

    float rgba[] = {
            0.f, 0.f, 0.f, .01f,
            0.f, 1.f, .5f, .25f,
            1.f, 1.f, 1.f, .5f,
            1.f, 1.f, .4f, .75f,
            1.f, .3f, .3f, 1.f
            };
    vkt::LookupTable lut(5,1,1,vkt::ColorFormat::RGBA32F);
    std::memcpy(lut.getData(), rgba, sizeof(rgba));

    vkt::RenderState renderState;
    renderState.renderAlgo = vkt::RenderAlgo::MultiScattering;
    renderState.rgbaLookupTable = lut.getResourceHandle();
    vkt::Render(volume, renderState);
}
