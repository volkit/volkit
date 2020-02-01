// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.h"
#include "forward.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    vktRenderAlgoRayMarching,
    vktRenderAlgoMultiScattering,
} vktRenderAlgo;

typedef struct
{
    //! Rendering algorithm
    vktRenderAlgo renderAlgo;

    //! Parameters related to ray marching algorithm
    ///@{

    //! Ray marching step size in object coordinates
    float dt;

    ///@}

    //! Parameters related to multi-scattering algorithm
    ///@{

    //! Majorant extinction coefficient
    float majorant;

    ///@}

    //! General parameters
    ///@{

    //! Width of the rendering viewport
    int viewportWidth;

    //! Height of the rendering viewport
    int viewportHeight;

    ///@}

} vktRenderState_t;

inline void vktRenderStateDefaultInit(vktRenderState_t* renderState)
{
#ifdef __cplusplus
    *renderState = {
        vktRenderAlgoRayMarching,
        1.f,
        1.f,
        512,
        512
        };
#else
    *renderState = (vktRenderState_t) {
        .renderAlgo = vktRenderAlgoRayMarching,
        .dt = 1.f,
        .majorant = 1.f,
        .viewportWidth = 512,
        .viewportHeight = 512
        };
#endif
}

VKTAPI vktError vktRenderSV(vktStructuredVolume volume,
                            vktRenderState_t renderState,
                            vktRenderState_t* newRenderState);

#ifdef __cplusplus
}
#endif
