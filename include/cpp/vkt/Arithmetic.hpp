// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

/*! @file  Arithmetic.hpp
 * @brief  Elementwise basic arithmetic operations
 *
 * Basic arithmetic operations performed per element, either on the whole
 * volume or on a range of the volume specified via `[first..last)`. Variants
 * of the operations prefixed with `Safe` check and prevent under and overflows
 * with respect to the minimal and maximal values that can be stored in the
 * volume.
 *
 * The operations are passed three volumes: the destination volume `dest`, as
 * well as two source volume `source1` and `source2`. It is generally allowable
 * that the reference to the destination volume points to the same memory as
 * the source volume (or, source volumes); however, when a destination `index
 * != 0` is specified, the behavior of the operation is undefined.
 * Specifically, with a parallel execution policy, it is not guaranteed that
 * the operations are performed atomically.
 */
namespace vkt
{
    //! Compute elementwise sum of two volumes
    VKTAPI Error Sum(StructuredVolume& dest,
                     StructuredVolume& source1,
                     StructuredVolume& source2);

    //! Compute elementwise sum over the range `[firstXXX..lastXXX)`
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

    //! Compute elementwise sum over the range `[first..last)`
    VKTAPI Error SumRange(StructuredVolume& dest,
                          StructuredVolume& source1,
                          StructuredVolume& source2,
                          Vec3i first,
                          Vec3i last,
                          Vec3i dstOffset = { 0, 0, 0 });

    //! Compute elementwise difference of two volumes
    VKTAPI Error Diff(StructuredVolume& dest,
                      StructuredVolume& source1,
                      StructuredVolume& source2);

    //! Compute elementwise difference over the range `[firstXXX..lastXXX)`
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

    //! Compute elementwise difference over the range `[first..last)`
    VKTAPI Error DiffRange(StructuredVolume& dest,
                           StructuredVolume& source1,
                           StructuredVolume& source2,
                           Vec3i first,
                           Vec3i last,
                           Vec3i dstOffset = { 0, 0, 0 });

    //! Compute elementwise product of two volumes
    VKTAPI Error Prod(StructuredVolume& dest,
                      StructuredVolume& source1,
                      StructuredVolume& source2);

    //! Compute elementwise product over the range `[firstXXX..lastXXX)`
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

    //! Compute elementwise product over the range `[first..last)`
    VKTAPI Error ProdRange(StructuredVolume& dest,
                           StructuredVolume& source1,
                           StructuredVolume& source2,
                           Vec3i first,
                           Vec3i last,
                           Vec3i dstOffset = { 0, 0, 0 });

    //! Compute elementwise quotient of two volumes
    VKTAPI Error Quot(StructuredVolume& dest,
                      StructuredVolume& source1,
                      StructuredVolume& source2);

    //! Compute elementwise quotient over the range `[firstXXX..lastXXX)`
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

    //! Compute elementwise quotient over the range `[first..last)`
    VKTAPI Error QuotRange(StructuredVolume& dest,
                           StructuredVolume& source1,
                           StructuredVolume& source2,
                           Vec3i first,
                           Vec3i last,
                           Vec3i dstOffset = { 0, 0, 0 });

    //! Compute elementwise abs. difference of two volumes
    VKTAPI Error AbsDiff(StructuredVolume& dest,
                         StructuredVolume& source1,
                         StructuredVolume& source2);

    //! Compute elementwise abs. difference over the range `[firstXXX..lastXXX)`
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

    //! Compute elementwise abs. difference over the range `[first..last)`
    VKTAPI Error AbsDiffRange(StructuredVolume& dest,
                              StructuredVolume& source1,
                              StructuredVolume& source2,
                              Vec3i first,
                              Vec3i last,
                              Vec3i dstOffset = { 0, 0, 0 });

    //! Compute safe sum of two volumes
    VKTAPI Error SafeSum(StructuredVolume& dest,
                         StructuredVolume& source1,
                         StructuredVolume& source2);

    //! Compute safe sum over the range `[firstXXX..lastXXX)`
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

    //! Compute safe sum over the range `[first..last)`
    VKTAPI Error SafeSumRange(StructuredVolume& dest,
                              StructuredVolume& source1,
                              StructuredVolume& source2,
                              Vec3i first,
                              Vec3i last,
                              Vec3i dstOffset = { 0, 0, 0 });

    //! Compute safe difference of two volumes
    VKTAPI Error SafeDiff(StructuredVolume& dest,
                          StructuredVolume& source1,
                          StructuredVolume& source2);

    //! Compute safe difference over the range `[firstXXX..lastXXX)`
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

    //! Compute safe difference over the range `[first..last)`
    VKTAPI Error SafeDiffRange(StructuredVolume& dest,
                               StructuredVolume& source1,
                               StructuredVolume& source2,
                               Vec3i first,
                               Vec3i last,
                               Vec3i dstOffset = { 0, 0, 0 });

    //! Compute safe product of two volumes
    VKTAPI Error SafeProd(StructuredVolume& dest,
                          StructuredVolume& source1,
                          StructuredVolume& source2);

    //! Compute safe product over the range `[firstXXX..lastXXX)`
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

    //! Compute safe product over the range `[first..last)`
    VKTAPI Error SafeProdRange(StructuredVolume& dest,
                               StructuredVolume& source1,
                               StructuredVolume& source2,
                               Vec3i first,
                               Vec3i last,
                               Vec3i dstOffset = { 0, 0, 0 });

    //! Compute safe quotient of two volumes
    VKTAPI Error SafeQuot(StructuredVolume& dest,
                          StructuredVolume& source1,
                          StructuredVolume& source2);

    //! Compute safe quotient over the range `[firstXXX..lastXXX)`
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

    //! Compute safe quotient over the range `[first..last)`
    VKTAPI Error SafeQuotRange(StructuredVolume& dest,
                               StructuredVolume& source1,
                               StructuredVolume& source2,
                               Vec3i first,
                               Vec3i last,
                               Vec3i dstOffset = { 0, 0, 0});

    //! Compute safe abs. difference of two volumes
    VKTAPI Error SafeAbsDiff(StructuredVolume& dest,
                             StructuredVolume& source1,
                             StructuredVolume& source2);

    //! Compute safe abs. difference over the range `[firstXXX..lastXXX)`
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

    //! Compute safe abs. difference over the range `[first..last)`
    VKTAPI Error SafeAbsDiffRange(StructuredVolume& dest,
                                  StructuredVolume& source1,
                                  StructuredVolume& source2,
                                  Vec3i first,
                                  Vec3i last,
                                  Vec3i dstOffset = { 0, 0, 0 });

} // vkt
