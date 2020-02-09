// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdint.h>

#include "common.h"
#include "forward.h"

#ifdef __cplusplus
extern "C"
{
#endif

VKTAPI vktError vktSumSV(vktStructuredVolume dest,
                         vktStructuredVolume source1,
                         vktStructuredVolume source2);

VKTAPI vktError vktSumRangeSV(vktStructuredVolume dest,
                              vktStructuredVolume source1,
                              vktStructuredVolume source2,
                              int32_t firstX,
                              int32_t firstY,
                              int32_t firstZ,
                              int32_t lastX,
                              int32_t lastY,
                              int32_t lastZ,
                              int32_t dstOffsetX,
                              int32_t dstOffsetY,
                              int32_t dstOffsetZ);

VKTAPI vktError vktDiffSV(vktStructuredVolume dest,
                          vktStructuredVolume source1,
                          vktStructuredVolume source2);

VKTAPI vktError vktDiffRangeSV(vktStructuredVolume dest,
                               vktStructuredVolume source1,
                               vktStructuredVolume source2,
                               int32_t firstX,
                               int32_t firstY,
                               int32_t firstZ,
                               int32_t lastX,
                               int32_t lastY,
                               int32_t lastZ,
                               int32_t dstOffsetX,
                               int32_t dstOffsetY,
                               int32_t dstOffsetZ);

VKTAPI vktError vktProdSV(vktStructuredVolume dest,
                          vktStructuredVolume source1,
                          vktStructuredVolume source2);

VKTAPI vktError vktProdRangeSV(vktStructuredVolume dest,
                               vktStructuredVolume source1,
                               vktStructuredVolume source2,
                               int32_t firstX,
                               int32_t firstY,
                               int32_t firstZ,
                               int32_t lastX,
                               int32_t lastY,
                               int32_t lastZ,
                               int32_t dstOffsetX,
                               int32_t dstOffsetY,
                               int32_t dstOffsetZ);

VKTAPI vktError vktQuotSV(vktStructuredVolume dest,
                          vktStructuredVolume source1,
                          vktStructuredVolume source2);

VKTAPI vktError vktQuotRangeSV(vktStructuredVolume dest,
                               vktStructuredVolume source1,
                               vktStructuredVolume source2,
                               int32_t firstX,
                               int32_t firstY,
                               int32_t firstZ,
                               int32_t lastX,
                               int32_t lastY,
                               int32_t lastZ,
                               int32_t dstOffsetX,
                               int32_t dstOffsetY,
                               int32_t dstOffsetZ);

VKTAPI vktError vktAbsDiffSV(vktStructuredVolume dest,
                             vktStructuredVolume source1,
                             vktStructuredVolume source2);

VKTAPI vktError vktAbsDiffRangeSV(vktStructuredVolume dest,
                                  vktStructuredVolume source1,
                                  vktStructuredVolume source2,
                                  int32_t firstX,
                                  int32_t firstY,
                                  int32_t firstZ,
                                  int32_t lastX,
                                  int32_t lastY,
                                  int32_t lastZ,
                                  int32_t dstOffsetX,
                                  int32_t dstOffsetY,
                                  int32_t dstOffsetZ);

VKTAPI vktError vktSafeSumSV(vktStructuredVolume dest,
                             vktStructuredVolume source1,
                             vktStructuredVolume source2);

VKTAPI vktError vktSafeSumRangeSV(vktStructuredVolume dest,
                                  vktStructuredVolume source1,
                                  vktStructuredVolume source2,
                                  int32_t firstX,
                                  int32_t firstY,
                                  int32_t firstZ,
                                  int32_t lastX,
                                  int32_t lastY,
                                  int32_t lastZ,
                                  int32_t dstOffsetX,
                                  int32_t dstOffsetY,
                                  int32_t dstOffsetZ);

VKTAPI vktError vktSafeDiffSV(vktStructuredVolume dest,
                              vktStructuredVolume source1,
                              vktStructuredVolume source2);

VKTAPI vktError vktSafeDiffRangeSV(vktStructuredVolume dest,
                                   vktStructuredVolume source1,
                                   vktStructuredVolume source2,
                                   int32_t firstX,
                                   int32_t firstY,
                                   int32_t firstZ,
                                   int32_t lastX,
                                   int32_t lastY,
                                   int32_t lastZ,
                                   int32_t dstOffsetX,
                                   int32_t dstOffsetY,
                                   int32_t dstOffsetZ);

VKTAPI vktError vktSafeProdSV(vktStructuredVolume dest,
                              vktStructuredVolume source1,
                              vktStructuredVolume source2);

VKTAPI vktError vktSafeProdRangeSV(vktStructuredVolume dest,
                                   vktStructuredVolume source1,
                                   vktStructuredVolume source2,
                                   int32_t firstX,
                                   int32_t firstY,
                                   int32_t firstZ,
                                   int32_t lastX,
                                   int32_t lastY,
                                   int32_t lastZ,
                                   int32_t dstOffsetX,
                                   int32_t dstOffsetY,
                                   int32_t dstOffsetZ);

VKTAPI vktError vktSafeQuotSV(vktStructuredVolume dest,
                              vktStructuredVolume source1,
                              vktStructuredVolume source2);

VKTAPI vktError vktSafeQuotRangeSV(vktStructuredVolume dest,
                                   vktStructuredVolume source1,
                                   vktStructuredVolume source2,
                                   int32_t firstX,
                                   int32_t firstY,
                                   int32_t firstZ,
                                   int32_t lastX,
                                   int32_t lastY,
                                   int32_t lastZ,
                                   int32_t dstOffsetX,
                                   int32_t dstOffsetY,
                                   int32_t dstOffsetZ);

VKTAPI vktError vktAbsDiffSV(vktStructuredVolume dest,
                             vktStructuredVolume source1,
                             vktStructuredVolume source2);

VKTAPI vktError vktAbsDiffRangeSV(vktStructuredVolume dest,
                                  vktStructuredVolume source1,
                                  vktStructuredVolume source2,
                                  int32_t firstX,
                                  int32_t firstY,
                                  int32_t firstZ,
                                  int32_t lastX,
                                  int32_t lastY,
                                  int32_t lastZ,
                                  int32_t dstOffsetX,
                                  int32_t dstOffsetY,
                                  int32_t dstOffsetZ);

#ifdef __cplusplus
}
#endif
