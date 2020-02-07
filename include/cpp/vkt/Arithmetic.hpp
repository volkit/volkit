// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    VKTAPI Error Sum(StructuredVolume& dest,
                     StructuredVolume& source1,
                     StructuredVolume& source2);

    VKTAPI Error SumRange(StructuredVolume& dest,
                          StructuredVolume& source1,
                          StructuredVolume& source2,
                          int32_t firstX,
                          int32_t firstY,
                          int32_t firstZ,
                          int32_t lastX,
                          int32_t lastY,
                          int32_t lastZ,
                          int32_t dstOffsetX = 0,
                          int32_t dstOffsetY = 0,
                          int32_t dstOffsetZ = 0);

    VKTAPI Error SumRange(StructuredVolume& dest,
                          StructuredVolume& source1,
                          StructuredVolume& source2,
                          Vec3i first,
                          Vec3i last,
                          Vec3i dstOffset = { 0, 0, 0 });

    VKTAPI Error Diff(StructuredVolume& dest,
                      StructuredVolume& source1,
                      StructuredVolume& source2);

    VKTAPI Error DiffRange(StructuredVolume& dest,
                           StructuredVolume& source1,
                           StructuredVolume& source2,
                           int32_t firstX,
                           int32_t firstY,
                           int32_t firstZ,
                           int32_t lastX,
                           int32_t lastY,
                           int32_t lastZ,
                           int32_t dstOffsetX = 0,
                           int32_t dstOffsetY = 0,
                           int32_t dstOffsetZ = 0);

    VKTAPI Error DiffRange(StructuredVolume& dest,
                           StructuredVolume& source1,
                           StructuredVolume& source2,
                           Vec3i first,
                           Vec3i last,
                           Vec3i dstOffset = { 0, 0, 0 });

    VKTAPI Error Prod(StructuredVolume& dest,
                      StructuredVolume& source1,
                      StructuredVolume& source2);

    VKTAPI Error ProdRange(StructuredVolume& dest,
                           StructuredVolume& source1,
                           StructuredVolume& source2,
                           int32_t firstX,
                           int32_t firstY,
                           int32_t firstZ,
                           int32_t lastX,
                           int32_t lastY,
                           int32_t lastZ,
                           int32_t dstOffsetX = 0,
                           int32_t dstOffsetY = 0,
                           int32_t dstOffsetZ = 0);

    VKTAPI Error ProdRange(StructuredVolume& dest,
                           StructuredVolume& source1,
                           StructuredVolume& source2,
                           Vec3i first,
                           Vec3i last,
                           Vec3i dstOffset = { 0, 0, 0 });

    VKTAPI Error Quot(StructuredVolume& dest,
                      StructuredVolume& source1,
                      StructuredVolume& source2);

    VKTAPI Error QuotRange(StructuredVolume& dest,
                           StructuredVolume& source1,
                           StructuredVolume& source2,
                           int32_t firstX,
                           int32_t firstY,
                           int32_t firstZ,
                           int32_t lastX,
                           int32_t lastY,
                           int32_t lastZ,
                           int32_t dstOffsetX = 0,
                           int32_t dstOffsetY = 0,
                           int32_t dstOffsetZ = 0);

    VKTAPI Error QuotRange(StructuredVolume& dest,
                           StructuredVolume& source1,
                           StructuredVolume& source2,
                           Vec3i first,
                           Vec3i last,
                           Vec3i dstOffset = { 0, 0, 0 });

    VKTAPI Error AbsDiff(StructuredVolume& dest,
                         StructuredVolume& source1,
                         StructuredVolume& source2);

    VKTAPI Error AbsDiffRange(StructuredVolume& dest,
                              StructuredVolume& source1,
                              StructuredVolume& source2,
                              int32_t firstX,
                              int32_t firstY,
                              int32_t firstZ,
                              int32_t lastX,
                              int32_t lastY,
                              int32_t lastZ,
                              int32_t dstOffsetX = 0,
                              int32_t dstOffsetY = 0,
                              int32_t dstOffsetZ = 0);

    VKTAPI Error AbsDiffRange(StructuredVolume& dest,
                              StructuredVolume& source1,
                              StructuredVolume& source2,
                              Vec3i first,
                              Vec3i last,
                              Vec3i dstOffset = { 0, 0, 0 });

    VKTAPI Error SafeSum(StructuredVolume& dest,
                         StructuredVolume& source1,
                         StructuredVolume& source2);

    VKTAPI Error SafeSumRange(StructuredVolume& dest,
                              StructuredVolume& source1,
                              StructuredVolume& source2,
                              int32_t firstX,
                              int32_t firstY,
                              int32_t firstZ,
                              int32_t lastX,
                              int32_t lastY,
                              int32_t lastZ,
                              int32_t dstOffsetX = 0,
                              int32_t dstOffsetY = 0,
                              int32_t dstOffsetZ = 0);

    VKTAPI Error SafeSumRange(StructuredVolume& dest,
                              StructuredVolume& source1,
                              StructuredVolume& source2,
                              Vec3i first,
                              Vec3i last,
                              Vec3i dstOffset = { 0, 0, 0 });

    VKTAPI Error SafeDiff(StructuredVolume& dest,
                          StructuredVolume& source1,
                          StructuredVolume& source2);

    VKTAPI Error SafeDiffRange(StructuredVolume& dest,
                               StructuredVolume& source1,
                               StructuredVolume& source2,
                               int32_t firstX,
                               int32_t firstY,
                               int32_t firstZ,
                               int32_t lastX,
                               int32_t lastY,
                               int32_t lastZ,
                               int32_t dstOffsetX = 0,
                               int32_t dstOffsetY = 0,
                               int32_t dstOffsetZ = 0);

    VKTAPI Error SafeDiffRange(StructuredVolume& dest,
                               StructuredVolume& source1,
                               StructuredVolume& source2,
                               Vec3i first,
                               Vec3i last,
                               Vec3i dstOffset = { 0, 0, 0 });

    VKTAPI Error SafeProd(StructuredVolume& dest,
                          StructuredVolume& source1,
                          StructuredVolume& source2);

    VKTAPI Error SafeProdRange(StructuredVolume& dest,
                               StructuredVolume& source1,
                               StructuredVolume& source2,
                               int32_t firstX,
                               int32_t firstY,
                               int32_t firstZ,
                               int32_t lastX,
                               int32_t lastY,
                               int32_t lastZ,
                               int32_t dstOffsetX = 0,
                               int32_t dstOffsetY = 0,
                               int32_t dstOffsetZ = 0);

    VKTAPI Error SafeProdRange(StructuredVolume& dest,
                               StructuredVolume& source1,
                               StructuredVolume& source2,
                               Vec3i first,
                               Vec3i last,
                               Vec3i dstOffset = { 0, 0, 0 });

    VKTAPI Error SafeQuot(StructuredVolume& dest,
                          StructuredVolume& source1,
                          StructuredVolume& source2);

    VKTAPI Error SafeQuotRange(StructuredVolume& dest,
                               StructuredVolume& source1,
                               StructuredVolume& source2,
                               int32_t firstX,
                               int32_t firstY,
                               int32_t firstZ,
                               int32_t lastX,
                               int32_t lastY,
                               int32_t lastZ,
                               int32_t dstOffsetX = 0,
                               int32_t dstOffsetY = 0,
                               int32_t dstOffsetZ = 0);

    VKTAPI Error SafeQuotRange(StructuredVolume& dest,
                               StructuredVolume& source1,
                               StructuredVolume& source2,
                               Vec3i first,
                               Vec3i last,
                               Vec3i dstOffset = { 0, 0, 0});

    VKTAPI Error SafeAbsDiff(StructuredVolume& dest,
                             StructuredVolume& source1,
                             StructuredVolume& source2);

    VKTAPI Error SafeAbsDiffRange(StructuredVolume& dest,
                                  StructuredVolume& source1,
                                  StructuredVolume& source2,
                                  int32_t firstX,
                                  int32_t firstY,
                                  int32_t firstZ,
                                  int32_t lastX,
                                  int32_t lastY,
                                  int32_t lastZ,
                                  int32_t dstOffsetX = 0,
                                  int32_t dstOffsetY = 0,
                                  int32_t dstOffsetZ = 0);

    VKTAPI Error SafeAbsDiffRange(StructuredVolume& dest,
                                  StructuredVolume& source1,
                                  StructuredVolume& source2,
                                  Vec3i first,
                                  Vec3i last,
                                  Vec3i dstOffset = { 0, 0, 0 });

} // vkt
