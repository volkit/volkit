#!/usr/bin/python3

import volkit as vkt

def main():
    # Volume dimensions
    dims = vkt.Vec3i()
    dims.x = 120
    dims.y = 66
    dims.z = 49

    # Brick size
    brickSize = vkt.Vec3i()
    brickSize.x = 16
    brickSize.y = 16
    brickSize.z = 16

    # Halo / ghost cells
    haloSizeNeg = vkt.Vec3i()
    haloSizeNeg.x = 1
    haloSizeNeg.y = 1
    haloSizeNeg.z = 1
    haloSizePos = vkt.Vec3i()
    haloSizePos.x = 1
    haloSizePos.y = 1
    haloSizePos.z = 1

    dataFormat = vkt.DataFormat_UInt8

    mappingLo = 0.
    mappingHi = 1.

    distX = 1.
    distY = 1.
    distZ = 1.

    volume = vkt.StructuredVolume(
            dims.x,
            dims.y,
            dims.z,
            dataFormat,
            distX,
            distY,
            distZ,
            mappingLo,
            mappingHi
            )

    # Put some values in
    vkt.Fill(volume, .1)

    # The destination data structure
    decomp = vkt.Array3D_StructuredVolume()

    # Preallocate storage for the decomposition
    vkt.BrickDecomposeResize(
            decomp,
            volume,
            brickSize,
            haloSizeNeg,
            haloSizePos
            )

    # Compute the decomposition
    vkt.BrickDecompose(
            decomp,
            volume,
            brickSize,
            haloSizeNeg,
            haloSizePos
            )

    vkt.Render(decomp[vkt.Vec3i()])

if __name__ == '__main__':
    main()
