#include <vkt/Fill.hpp>
#include <vkt/Render.hpp>
#include <vkt/Scan.hpp>
#include <vkt/StructuredVolume.hpp>

#include "common.h"

int main()
{
    // Data format
    vkt::DataFormat dataFormat = vkt::DataFormat::UInt8;

    // Mapping for highest/lowest voxel value
    float mappingLo = 0.f;
    float mappingHi = 1.f;

    // Voxel distance
    float distX = 1.f;
    float distY = 1.f;
    float distZ = 1.f;


    //--- Create a volume ---------------------------------

    vkt::StructuredVolume volume(8, 8, 8,
                                 dataFormat,
                                 distX,
                                 distY,
                                 distZ,
                                 mappingLo,
                                 mappingHi);

    // Fill the volume
    VKT_SAFE_CALL(vkt::Fill(volume, .02f));

    //--- ScanRange ---------------------------------------

    // Note how dst and src are the same
    VKT_SAFE_CALL(vkt::ScanRange(volume, // dst
                                 volume, // src
                                 0, 0, 0,
                                 4, 4, 4,
                                 0, 0, 0));

    // In the following, some components of first > last
    VKT_SAFE_CALL(vkt::ScanRange(volume,
                                 volume,
                                 7, 0, 0,
                                 3, 4, 4,
                                 0, 0, 0));

    VKT_SAFE_CALL(vkt::ScanRange(volume,
                                 volume,
                                 0, 7, 0,
                                 4, 3, 4,
                                 0, 0, 0));

    VKT_SAFE_CALL(vkt::ScanRange(volume,
                                 volume,
                                 0, 0, 7,
                                 4, 4, 3,
                                 0, 0, 0));

    VKT_SAFE_CALL(vkt::ScanRange(volume,
                                 volume,
                                 7, 7, 0,
                                 3, 3, 4,
                                 0, 0, 0));

    VKT_SAFE_CALL(vkt::ScanRange(volume,
                                 volume,
                                 0, 7, 7,
                                 4, 3, 3,
                                 0, 0, 0));

    VKT_SAFE_CALL(vkt::ScanRange(volume,
                                 volume,
                                 7, 0, 7,
                                 3, 4, 3,
                                 0, 0, 0));

    VKT_SAFE_CALL(vkt::ScanRange(volume,
                                 volume,
                                 7, 7, 7,
                                 3, 3, 3,
                                 0, 0, 0));

    //--- Render ------------------------------------------

    // Render volume
    VKT_SAFE_CALL(vkt::Render(volume));
}
