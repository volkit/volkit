// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdint.h>

#include <cuda_runtime_api.h>

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

struct vktCudaContext_impl;

typedef struct vktCudaContext_impl* vktCudaContext;

VKTAPI vktError vktCudaContextCreate(vktCudaContext* context);

VKTAPI vktError vktCudaContextDestroy(vktCudaContext context);

/*!
 * @brief  set execution mode for algorithms to async
 *
 * @param  async (in) [true|false]
 */
VKTAPI vktError vktCudaContextSetAsyncExecution(vktCudaContext context,
                                                int async);

/*!
 * @brief  get execution mode for algorithms
 *
 * @param  async (out) [true|false]
 */
VKTAPI vktError vktCudaContextGetAsyncExecution(vktCudaContext context,
                                                int* async);

VKTAPI vktError vktCudaContextSetNumStreams(vktCudaContext context,
                                            int32_t numStreams);

VKTAPI vktError vktCudaContextGetNumStreams(vktCudaContext context,
                                            int32_t* numStreams);

VKTAPI vktError vktCudaContextSetStream(vktCudaContext context,
                                        int32_t streamId,
                                        cudaStream_t stream);

VKTAPI vktError vktCudaContextGetStream(vktCudaContext context,
                                        int32_t streamId,
                                        cudaStream_t* stream);

VKTAPI vktError vktCudaContextSetComputeStreamId(vktCudaContext context,
                                                 int32_t streamId);

VKTAPI vktError vktCudaContextGetComputeStreamId(vktCudaContext context,
                                                 int32_t* streamId);

VKTAPI vktError vktCudaContextSetCopyStreamId(vktCudaContext context,
                                              int32_t streamId);

VKTAPI vktError vktCudaContextGetCopyStreamId(vktCudaContext context,
                                              int32_t* streamId);

#ifdef __cplusplus
}
#endif
