#include <vkt/Decompose.hpp>
#include <vkt/linalg.hpp>
#include <vkt/Fill.hpp>
#include <vkt/Render.hpp>
#include <vkt/StructuredVolume.hpp>

int main()
{
    // Volume dimensions
    vkt::Vec3i dims(256,256,256);

    // Brick size
    vkt::Vec3i brickSize(16);

    // Halo / ghost cells
    vkt::Vec3i haloSizeNeg(1,1,1);
    vkt::Vec3i haloSizePos(1,1,1);

    int bpv = 1;

    float mappingLo = 0.f;
    float mappingHi = 1.f;

    float distX = 1.f;
    float distY = 1.f;
    float distZ = 1.f;

    vkt::StructuredVolume volume(
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

    vkt::FillRange(
            volume,
            vkt::Vec3i(0,0,0),
            dims,
            .02f
            );

    // Query the number of bricks the volume will be decomposed into
    vkt::Vec3i numBricks;
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
            distX,
            distY,
            distZ,
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
            decomp[vkt::Vec3i(0,0,0)],
            {}
            );
}
