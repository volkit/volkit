#include <vkt/Decompose.hpp>
#include <vkt/linalg.hpp>
#include <vkt/Fill.hpp>
#include <vkt/Render.hpp>
#include <vkt/StructuredVolume.hpp>

int main()
{
    // Volume dimensions
    vkt::vec3i dims(256,256,256);

    // Brick size
    vkt::vec3i brickSize(16);

    // Halo / ghost cells
    vkt::vec3i haloSizeNeg(1,1,1);
    vkt::vec3i haloSizePos(1,1,1);

    int bpv = 1;

    float mappingLo = 0.f;
    float mappingHi = 1.f;

    vkt::StructuredVolume volume(
            dims.x,
            dims.y,
            dims.z,
            bpv,
            mappingLo,
            mappingHi
            );

    vkt::FillRange(
            volume,
            vkt::vec3i(0,0,0),
            dims,
            .02f
            );

    // Query the number of bricks the volume will be decomposed into
    vkt::vec3i numBricks;
    vkt::BrickDecomposeGetNumBricks(
            numBricks,
            dims,
            brickSize,
            haloSizeNeg,
            haloSizePos
            );

    // Allocate storage for numBricks bricks with halos
    vkt::Array3D<vkt::StructuredVolume> decomp(numBricks);
    for (vkt::StructuredVolume& vol : decomp)
    {
        vol = vkt::StructuredVolume(
            haloSizeNeg.x + brickSize.x + haloSizePos.x,
            haloSizeNeg.y + brickSize.y + haloSizePos.y,
            haloSizeNeg.z + brickSize.z + haloSizePos.z,
            bpv,
            mappingLo,
            mappingHi
            );
    }

    vkt::BrickDecompose(
            decomp,
            volume,
            brickSize,
            haloSizeNeg,
            haloSizePos
            );

    vkt::Render(
            decomp[vkt::vec3i(0,0,0)],
            {}
            );
}
