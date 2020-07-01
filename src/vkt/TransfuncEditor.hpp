// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <GL/glew.h>

#include <vkt/LookupTable.hpp>
#include <vkt/ManagedResource.hpp>

namespace vkt
{
    class TransfuncEditor
    {
    public:
       ~TransfuncEditor();

        //! Set a user-provided LUT that a copy is created from
        void setLookupTableResource(ResourceHandle handle);

        //! Get an updated LUT that is a copied of the user-provided one
        LookupTable* getUpdatedLookupTable() const;

        //! Set a zoom range to visually zoom into the LUT
        void setZoom(float min, float max);

        //! Indicates that the internal copy of the LUT has changed
        bool updated() const;

        //! Render with ImGui
        void show();

    private:
        // Local LUT copy
        LookupTable* rgbaLookupTable_ = nullptr;

        // User-provided LUT
        LookupTable* userLookupTable_ = nullptr;

        // Zoom min set by user
        float zoomMin_ = 0.f;

        // Zoom max set by user
        float zoomMax_ = 1.f;

        // Flag indicating that texture needs to be regenerated
        bool lutChanged_ = false;

        // RGB texture
        GLuint texture_ = GLuint(-1);

        // Drawing canvas size
        Vec2i canvasSize_ = { 300, 150 };

        // Mouse state for drawing
        struct MouseEvent
        {
            enum Type { PassiveMotion, Motion, Press, Release };
            enum Button { Left, Middle, Right, None };

            Vec2i pos = { 0, 0 };
            int button = None;
            Type type = Motion;
        };

        // The last mouse event
        MouseEvent lastEvent_;

        // Drawing in progress
        bool drawing_ = false;

        // Raster LUT to image and upload with OpenGL
        void rasterTexture();

        // Generate mouse event when mouse hovered over rect
        MouseEvent generateMouseEvent();

        // Handle mouse event
        void handleMouseEvent(MouseEvent const& event);
    };

} // vkt
