// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include <imgui.h>

#include "ColorFormatInfo.hpp"
#include "TransfuncEditor.hpp"

#include "linalg.hpp"

static void enableBlendCB(ImDrawList const*, ImDrawCmd const*)
{
    glEnable(GL_BLEND);
}

static void disableBlendCB(ImDrawList const*, ImDrawCmd const*)
{
    glDisable(GL_BLEND);
}

namespace vkt
{
    TransfuncEditor::~TransfuncEditor()
    {
        // TODO: cannot be sure we have a GL context here!
        glDeleteTextures(1, &texture_);

        delete rgbaLookupTable_;
    }

    void TransfuncEditor::setLookupTableResource(ResourceHandle handle)
    {
        lutChanged_ = true;

        userLookupTable_ = (LookupTable*)GetManagedResource(handle);
    }

    LookupTable* TransfuncEditor::getUpdatedLookupTable() const
    {
        return rgbaLookupTable_;
    }

    bool TransfuncEditor::updated() const
    {
        return lutChanged_;
    }

    void TransfuncEditor::show()
    {
        if (userLookupTable_ == nullptr)
            return;

        ImGui::Begin("TransfuncEditor");

        rasterTexture();

        ImGui::GetWindowDrawList()->AddCallback(disableBlendCB, nullptr);
        ImGui::ImageButton(
            (void*)(intptr_t)texture_,
            ImVec2(canvasSize_.x, canvasSize_.y),
            ImVec2(0, 0),
            ImVec2(1, 1),
            0 // frame size = 0
            );

        MouseEvent event = generateMouseEvent();
        handleMouseEvent(event);

        ImGui::GetWindowDrawList()->AddCallback(enableBlendCB, nullptr);

        ImGui::End();
    }

    void TransfuncEditor::rasterTexture()
    {
        if (!lutChanged_)
            return;

        // TODO: maybe move to a function called by user?
        if (texture_ == GLuint(-1))
        {
            glGenTextures(1, &texture_);
            glBindTexture(GL_TEXTURE_2D, texture_);
        }

        Vec3i dims = userLookupTable_->getDims();
        ColorFormat format = userLookupTable_->getColorFormat();

        if (dims.x >= 1 && dims.y == 1 && dims.z == 1 && format == ColorFormat::RGBA32F)
        {
            float* colors = nullptr;
            float* updated = nullptr;

            if (rgbaLookupTable_ == nullptr)
            {
                rgbaLookupTable_ = new LookupTable(canvasSize_.x, 1, 1, userLookupTable_->getColorFormat());

                // The user-provided colors
                colors = (float*)userLookupTable_->getData();

                // Updated colors
                updated = (float*)rgbaLookupTable_->getData();

                // Lerp colors and alpha
                for (int i = 0; i < canvasSize_.x; ++i)
                {
                    float indexf = i / (float)(canvasSize_.x - 1) * (dims.x - 1);
                    int indexa = (int)indexf;
                    int indexb = min(indexa + 1, dims.x - 1);
                    Vec3f rgb1{ colors[4 * indexa], colors[4 * indexa + 1], colors[4 * indexa + 2] };
                    float alpha1 = colors[4 * indexa + 3];
                    Vec3f rgb2{ colors[4 * indexb], colors[4 * indexb + 1], colors[4 * indexb + 2] };
                    float alpha2 = colors[4 * indexb + 3];
                    float frac = indexf - indexa;

                    Vec3f rgb = lerp(rgb1, rgb2, frac);
                    float alpha = lerp(alpha1, alpha2, frac);

                    updated[4 * i]     = rgb.x;
                    updated[4 * i + 1] = rgb.y;
                    updated[4 * i + 2] = rgb.z;
                    updated[4 * i + 3] = alpha;
                }
            }
            else
            {
                // The user-provided colors
                colors = (float*)userLookupTable_->getData();

                // Updated colors
                updated = (float*)rgbaLookupTable_->getData();

            }

            // Blend on the CPU (TODO: figure out how textures can be
            // blended with ImGui..)
            std::vector<Vec4f> rgba(canvasSize_.x * canvasSize_.y);
            for (int y = 0; y < canvasSize_.y; ++y)
            {
                for (int x = 0; x < canvasSize_.x; ++x)
                {
                    Vec3f rgb{ updated[4 * x], updated[4 * x + 1], updated[4 * x + 2] };
                    float alpha = updated[4 * x + 3];

                    float grey = .9f;
                    float a = ((canvasSize_.y - y - 1) / (float)canvasSize_.y) <= alpha ? .6f : 0.f;

                    rgba[y * canvasSize_.x + x].x = (1 - a) * rgb.x + a * grey;
                    rgba[y * canvasSize_.x + x].y = (1 - a) * rgb.y + a * grey;
                    rgba[y * canvasSize_.x + x].z = (1 - a) * rgb.z + a * grey;
                    rgba[y * canvasSize_.x + x].w = 1.f;
                }
            }

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                ColorFormatInfoTableGL[(int)format].internalFormat,
                canvasSize_.x,
                canvasSize_.y,
                0,
                ColorFormatInfoTableGL[(int)format].format,
                ColorFormatInfoTableGL[(int)format].type,
                rgba.data()
                );
        }
        else
            assert(0);

        lutChanged_ = false;
    }

    TransfuncEditor::MouseEvent TransfuncEditor::generateMouseEvent()
    {
        MouseEvent event;

        int x = ImGui::GetIO().MousePos.x - ImGui::GetCursorScreenPos().x;
        int y = ImGui::GetCursorScreenPos().y - ImGui::GetIO().MousePos.y - 1;
        
        event.pos = { x, y };
        event.button = ImGui::GetIO().MouseDown[0] ? MouseEvent::Left :
                       ImGui::GetIO().MouseDown[1] ? MouseEvent::Middle :
                       ImGui::GetIO().MouseDown[2] ? MouseEvent::Right:
                                                     MouseEvent::None;
        // TODO: handle the unlikely case that the down button is not
        // the same as the one from lastEvent_. This could happen as
        // the mouse events are tied to the rendering frame rate
        if (event.button == MouseEvent::None && lastEvent_.button == MouseEvent::None)
            event.type = MouseEvent::PassiveMotion;
        else if (event.button != MouseEvent::None && lastEvent_.button != MouseEvent::None)
            event.type = MouseEvent::Motion;
        else if (event.button != MouseEvent::None && lastEvent_.button == MouseEvent::None)
            event.type = MouseEvent::Press;
        else
            event.type = MouseEvent::Release;

        return event;
    }

    void TransfuncEditor::handleMouseEvent(TransfuncEditor::MouseEvent const& event)
    {
        bool hovered = ImGui::IsItemHovered()
                && event.pos.x >= 0 && event.pos.x < canvasSize_.x
                && event.pos.y >= 0 && event.pos.y < canvasSize_.y;

        if (event.type == MouseEvent::PassiveMotion || event.type == MouseEvent::Release)
            drawing_ = false;

        if (drawing_ || (event.type == MouseEvent::Press && hovered && event.button == MouseEvent::Left))
        {
            float* updated = (float*)rgbaLookupTable_->getData();

            // Allow for drawing even when we're slightly outside
            // (i.e. not hovering) the drawing area
            int thisX = clamp(event.pos.x, 0, canvasSize_.x - 1);
            int thisY = clamp(event.pos.y, 0, canvasSize_.y - 1);
            int lastX = clamp(lastEvent_.pos.x, 0, canvasSize_.x - 1);

            updated[4 * thisX + 3] = thisY / (float)(canvasSize_.y - 1);

            // Also set the alphas that were potentially skipped b/c
            // the mouse movement was faster than the rendering frame
            // rate
            if (lastEvent_.button == MouseEvent::Left
                && std::abs(lastX - thisX) > 1)
            {
                float alpha1;
                float alpha2;
                if (lastX > thisX)
                {
                    alpha1 = updated[4 * lastX + 3];
                    alpha2 = updated[4 * thisX + 3];
                }
                else
                {
                    alpha1 = updated[4 * thisX + 3];
                    alpha2 = updated[4 * lastX + 3];
                }

                int inc = lastEvent_.pos.x < event.pos.x ? 1 : -1;

                for (int x = lastX + inc; x != thisX; x += inc)
                {
                    float frac = (thisX - x) / (float)std::abs(thisX - lastX);

                    updated[4 * x + 3] = lerp(alpha1, alpha2, frac);
                }
            }

            lutChanged_ = true;
            drawing_ = true;
        }

        lastEvent_ = event;
    }

} // vkt
