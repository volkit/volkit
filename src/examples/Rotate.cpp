#include <cmath>

#include <vkt/Fill.hpp>
#include <vkt/Filter.hpp>
#include <vkt/LookupTable.hpp>
#include <vkt/Render.hpp>
#include <vkt/Rotate.hpp>
#include <vkt/StructuredVolume.hpp>

int main()
{
    // Create a structured volume
    vkt::Vec3i dims = { 256,128,100 };

    vkt::DataFormat dataFormat = vkt::DataFormat::UInt8;
    vkt::StructuredVolume volume(
            dims.x, dims.y, dims.z,
            dataFormat,
            1.f, 1.f, 1.f, // dist
            0.f, 1.f // mapping
            );

    vkt::Fill(volume, .1f);

    vkt::FillRange(
            volume,
            { 64,4,4 },
            { 192,124,96 },
            1.f
            );

    // Destination volume; has the same size as the original one
    vkt::StructuredVolume rotatedVolume(
            dims.x, dims.y, dims.z,
            dataFormat,
            1.f, 1.f, 1.f,
            0.f, 1.f
            );

    vkt::Fill(rotatedVolume, .1f);

    // Rotate the volume with rotation center in the middle
    vkt::Rotate(
            rotatedVolume,
            volume,
            { 1.f,.3f,0.f },                     // rotation axis
            45.f * M_PI / 180.f,                 // rotation angle in radians
            { dims.x*.5f,dims.y*.5f,dims.z*.5f } // center of rotation
            );

    // float data[] = {
    //    -1.f, -3.f, -1.f,
    //    -3.f, -6.f, -3.f,
    //    -1.f, -3.f, -1.f,
    //     0.f, 0.f, 0.f,
    //     0.f, 0.f, 0.f,
    //     0.f, 0.f, 0.f,
    //     1.f, 3.f, 1.f,
    //     3.f, 6.f, 3.f,
    //     1.f, 3.f, 1.f
    // };
    float data[] = {
        1.f, 3.f, 1.f,
        3.f, 5.f, 3.f,
        1.f, 3.f, 1.f,

        3.f, 5.f, 3.f,
        5.f, 7.f, 5.f,
        3.f, 5.f, 3.f,

        1.f, 3.f, 1.f,
        3.f, 5.f, 3.f,
        1.f, 3.f, 1.f
    };
    float sum = 0.f;
    for (int i=0; i<27; ++i) sum += data[i];
    for (int i=0; i<27; ++i) data[i] /= sum*1.5f;
    vkt::Filter sobel(data, vkt::Vec3i{3,3,3});
    vkt::ApplyFilter(rotatedVolume,rotatedVolume,sobel,vkt::AddressMode::Border);

    float rgba[] = {
            1.f, 1.f, 1.f, .005f,
            0.f, .1f, .1f, .25f,
            .5f, .5f, .7f, .5f,
            .7f, .7f, .07f, 1.f,
            1.f, .3f, .3f, 1.f
            };
    vkt::LookupTable lut(5,1,1,vkt::ColorFormat::RGBA32F);
    lut.setData((uint8_t*)rgba);

    vkt::RenderState renderState;
    renderState.renderAlgo = vkt::RenderAlgo::MultiScattering;
    renderState.rgbaLookupTable = lut.getResourceHandle();
    renderState.viewportWidth = 800;
    renderState.viewportHeight = 800;
    renderState.initialCamera.isSet = 1;
    renderState.initialCamera.eye = {-148.362,147.885,215.525};
    renderState.initialCamera.center = {107.208,52.5183,33.4578};
    renderState.initialCamera.up = {0.244828,0.956698,-0.157449};
    vkt::Render(rotatedVolume, renderState);
}
