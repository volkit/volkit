// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/ManagedResource.h>

#include "common.h"
#include "forward.h"
#include "linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    vktRenderAlgoRayMarching,
    vktRenderAlgoImplicitIso,
    vktRenderAlgoMultiScattering,
} vktRenderAlgo;

typedef struct
{
    //! Rendering algorithm
    vktRenderAlgo renderAlgo;

    //! Parameters related to ray marching algorithm
    ///@{

    //! Ray marching step size in object coordinates
    float dtRayMarching;

    ///@}

    //! Parameters related to implicit iso algorithm
    ///@{

    //! The number of activated iso values
    uint16_t numIsoSurfaces;

    //! The maximum number of iso surfaces
    enum { MaxIsoSurfaces = 10 };

    //! The iso surfaces
    float isoSurfaces[MaxIsoSurfaces];

    //! Implicit iso step size in object coordinates
    float dtImplicitIso;

    ///@}

    //! Parameters related to multi-scattering algorithm
    ///@{

    //! Majorant extinction coefficient
    float majorant;

    ///@}

    //! Parameters related to animation
    ///@{

    unsigned animationFrame;

    ///@}

    //! General parameters
    ///@{

    //! RGBA32F lookup table for absorption, emission and scattering albedo
    vktResourceHandle rgbaLookupTable;

    //! Histogram, will be displayed in a widget inside the viewport
    vktResourceHandle histogram;

    //! Width of the rendering viewport
    int viewportWidth;

    //! Height of the rendering viewport
    int viewportHeight;

    //! Convert final colors from linear to sRGB
    vktBool_t sRGB;

    //! Initial camera, optionally set by the user
    struct
    {
        //! By default, this isn't set but determined using viewAll()
        vktBool_t isSet;

        //! Position where the camera is at
        vktVec3f_t eye;

        //! Position that we're looking at
        vktVec3f_t center;

        //! Camera up vector
        vktVec3f_t up;

        //! vertical field of view, specified in degree
        float fovy;

        //! Lens radius, used for depth of field
        float lensRadius;

        //! Distance we're focusing at, used for depth of field
        float focalDistance;
    } initialCamera;

    //! Take snapshots using a key or when the viewer was closed
    struct
    {
        //! Not enabled by default
        vktBool_t enabled;

        //! File to store the snap shot to. Ending determines file type
        char const* fileName;

        //! Overrides key press
        vktBool_t takeOnClose;

        //! If @see takeOnClose not set, this key is used
        char key;

        //! Optional message that is printed to the console
        char const* message;
    } snapshotTool;
    ///@}

} vktRenderState_t;

static void vktRenderStateDefaultInit(vktRenderState_t* renderState)
{
#ifdef __cplusplus
    *renderState = {
        vktRenderAlgoRayMarching,
        1.f,
        1,
        { .5f },
        1.f,
        1.f,
        0,
        vktResourceHandle(-1),
        vktResourceHandle(-1),
        512,
        512,
        1,
        { 0, { 0.f, 0.f, 0.f }, { 0.f, 0.f, -1.f }, { 0.f, 1.f, 0.f }, 45.f, .001f, 10.f },
        { 0, "", 0, 'p', "" }
        };
#else
    *renderState = (vktRenderState_t) {
        .renderAlgo = vktRenderAlgoRayMarching,
        .dtRayMarching = 1.f,
        .numIsoSurfaces = 1,
        .isoSurfaces = { .5f },
        .dtImplicitIso = 1.f,
        .majorant = 1.f,
        .animationFrame = 0,
        .rgbaLookupTable = -1,
        .histogram = -1,
        .viewportWidth = 512,
        .viewportHeight = 512,
        .sRGB = 1,
        .initialCamera = { 0, { 0.f, 0.f, 0.f }, { 0.f, 0.f, -1.f }, { 0.f, 1.f, 0.f }, 45.f, .001f, 10.f },
        .snapshotTool = { 0, "", 0, 'p', "" }
        };
#endif
}

VKTAPI vktError vktRenderSV(vktStructuredVolume volume,
                            vktRenderState_t renderState,
                            vktRenderState_t* newRenderState);

#ifdef __cplusplus
}
#endif
