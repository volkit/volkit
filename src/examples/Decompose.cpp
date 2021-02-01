#include <vkt/Decompose.hpp>
#include <vkt/linalg.hpp>
#include <vkt/Fill.hpp>
#include <vkt/Render.hpp>
#include <vkt/StructuredVolume.hpp>

int main()
{
    // Volume dimensions
    vkt::Vec3i dims = { 120,66,49 };

    // Brick size
    vkt::Vec3i brickSize = { 16, 16, 16 };

    // Halo / ghost cells
    vkt::Vec3i haloSizeNeg = { 1, 1, 1 };
    vkt::Vec3i haloSizePos = { 1, 1, 1 };

    vkt::DataFormat dataFormat = vkt::DataFormat::UInt8;

    float mappingLo = 0.f;
    float mappingHi = 1.f;

    float distX = 1.f;
    float distY = 1.f;
    float distZ = 1.f;

    vkt::StructuredVolume volume(
            dims.x,
            dims.y,
            dims.z,
            dataFormat,
            distX,
            distY,
            distZ,
            mappingLo,
            mappingHi
            );

    // Put some values in
    vkt::Fill(volume, .1f);

    // The destination data structure
    vkt::Array3D<vkt::StructuredVolume> decomp;

    // Preallocate storage for the decomposition
    vkt::BrickDecomposeResize(
            decomp,
            volume,
            brickSize,
            haloSizeNeg,
            haloSizePos
            );

    // Compute the decomposition
    vkt::BrickDecompose(
            decomp,
            volume,
            brickSize,
            haloSizeNeg,
            haloSizePos
            );

    vkt::Render(decomp[{0,0,0}]);
}
