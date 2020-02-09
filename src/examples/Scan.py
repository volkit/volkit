#!/usr/bin/python3

import volkit as vkt

def main():

    #--- Create a volume ----------------------------------

    bpv = 1
    mappingLo = 0.
    mappingHi = 1.
    distX = 1.
    distY = 1.
    distZ = 1.

    volume = vkt.StructuredVolume(8, 8, 8,
                                  bpv,
                                  distX,
                                  distY,
                                  distZ,
                                  mappingLo,
                                  mappingHi)

    # Fill the volume
    vkt.Fill(volume, .02)

    #--- ScanRange-----------------------------------------

    # Note how dst and src are the same
    vkt.ScanRange(volume, # dst
                  volume, # src
                  0, 0, 0,
                  4, 4, 4,
                  0, 0, 0)

    # In the following, some components of first > last
    vkt.ScanRange(volume,
                  volume,
                  7, 0, 0,
                  3, 4, 4,
                  0, 0, 0)

    vkt.ScanRange(volume,
                  volume,
                  0, 7, 0,
                  4, 3, 4,
                  0, 0, 0)

    vkt.ScanRange(volume,
                  volume,
                  0, 0, 7,
                  4, 4, 3,
                  0, 0, 0)

    vkt.ScanRange(volume,
                  volume,
                  7, 7, 0,
                  3, 3, 4,
                  0, 0, 0)

    vkt.ScanRange(volume,
                  volume,
                  0, 7, 7,
                  4, 3, 3,
                  0, 0, 0)

    vkt.ScanRange(volume,
                  volume,
                  7, 0, 7,
                  3, 4, 3,
                  0, 0, 0)

    vkt.ScanRange(volume,
                  volume,
                  7, 7, 7,
                  3, 3, 3,
                  0, 0, 0)

    #--- Render -------------------------------------------

    # Render volume
    renderState = vkt.RenderState()
    vkt.Render(volume, renderState)

if __name__ == '__main__':
    main()
