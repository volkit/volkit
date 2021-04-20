#include <cstdlib>
#include <iostream>
#include <ostream>

#include <vkt/ExecutionPolicy.hpp>
#include <vkt/InputStream.hpp>
#include <vkt/Histogram.hpp>
#include <vkt/LookupTable.hpp>
#include <vkt/Render.hpp>
#include <vkt/Resample.hpp>
#include <vkt/StructuredVolume.hpp>
#include <vkt/VolumeFile.hpp>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " file.raw\n";
        return EXIT_FAILURE;
    }

    // Switch performance debugging on
    vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
    ep.printPerformance = vkt::True;
    vkt::SetThreadExecutionPolicy(ep);

    vkt::VolumeFile file(argv[1], vkt::OpenMode::Read);

    vkt::VolumeFileHeader header = file.getHeader();

    if (!header.isStructured)
    {
        std::cerr << "No valid volume file\n";
        return EXIT_FAILURE;
    }

    vkt::Vec3i dims = header.dims;
    if (dims.x * dims.y * dims.z < 1)
    {
        std::cerr << "Cannot parse dimensions from file name\n";
        return EXIT_FAILURE;
    }

    vkt::DataFormat dataFormat = header.dataFormat;
    if (dataFormat == vkt::DataFormat::Unspecified)
    {
        std::cerr << "Cannot parse data format from file name, guessing uint8...\n";
        dataFormat = vkt::DataFormat::UInt8;
    }

    vkt::StructuredVolume volume(dims.x, dims.y, dims.z, dataFormat);
    vkt::InputStream is(file);
    is.read(volume);

    // Resample using contrast limited adaptive histogram equalization
    // Source volume is also dest volume
    vkt::ResampleCLAHE(volume, volume);

    float rgba[] = {
            1.f, 1.f, 1.f, .005f,
            0.f, .1f, .1f, .25f,
            .5f, .5f, .7f, .5f,
            .7f, .7f, .07f, .75f,
            1.f, .3f, .3f, 1.f
            };
    vkt::LookupTable lut(5,1,1,vkt::ColorFormat::RGBA32F);
    lut.setData((uint8_t*)rgba);

    // Compute histogram
    vkt::Histogram histogram(256);
    vkt::ComputeHistogram(volume, histogram);

    vkt::RenderState renderState;
    //renderState.renderAlgo = vkt::RenderAlgo::RayMarching;
    renderState.renderAlgo = vkt::RenderAlgo::ImplicitIso;
    renderState.renderAlgo = vkt::RenderAlgo::MultiScattering;
    renderState.rgbaLookupTable = lut.getResourceHandle();
    // Set handle to histogram to make visible in renderer
    renderState.histogram = histogram.getResourceHandle();
    vkt::Render(volume, renderState);
}
