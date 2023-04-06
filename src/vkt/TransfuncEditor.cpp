// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include <GL/glew.h>

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
        delete normalizedHistogram_;
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

    void TransfuncEditor::setHistogramResource(ResourceHandle handle)
    {
        histogramChanged_ = true;

        userHistogram_ = (Histogram*)GetManagedResource(handle);
    }

    void TransfuncEditor::setZoom(float min, float max)
    {
        zoomMin_ = min;
        zoomMax_ = max;
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

        drawImmediate();

        ImGui::End();
    }

    void TransfuncEditor::drawImmediate()
    {
        if (userLookupTable_ == nullptr)
            return;

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
    }

    void TransfuncEditor::rasterTexture()
    {
        if (!lutChanged_ && !histogramChanged_)
            return;

        setExecutionPolicyCPU();

        if (histogramChanged_)
            normalizeHistogram();

        // TODO: maybe move to a function called by user?
        if (texture_ == GLuint(-1))
            glGenTextures(1, &texture_);

        GLint prevTexture;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTexture);
        glBindTexture(GL_TEXTURE_2D, texture_);

        Vec3i userDims = userLookupTable_->getDims();
        ColorFormat format = userLookupTable_->getColorFormat();

        if (userDims.x >= 1 && userDims.y == 1 && userDims.z == 1 && format == ColorFormat::RGBA32F)
        {
            float* colors = nullptr;
            float* updated = nullptr;

            if (rgbaLookupTable_ == nullptr)
            {
                rgbaLookupTable_ = new LookupTable(8192, 1, 1, userLookupTable_->getColorFormat());

                Vec3i actualDims = rgbaLookupTable_->getDims();

                // The user-provided colors
                colors = (float*)userLookupTable_->getData();

                // Updated colors
                updated = (float*)rgbaLookupTable_->getData();

                // Lerp colors and alpha
                for (int i = 0; i < actualDims.x; ++i)
                {
                    float indexf = i / (float)(actualDims.x - 1) * (userDims.x - 1);
                    int indexa = (int)indexf;
                    int indexb = min(indexa + 1, userDims.x - 1);
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
            Vec3i actualDims = rgbaLookupTable_->getDims();
            for (int y = 0; y < canvasSize_.y; ++y)
            {
                for (int x = 0; x < canvasSize_.x; ++x)
                {
                    float indexf = x / (float)(canvasSize_.x - 1);
                    indexf *= zoomMax_ - zoomMin_;
                    indexf += zoomMin_;
                    indexf *= actualDims.x - 1;
                    int xx = (int)indexf;

                    Vec3f rgb{ updated[4 * xx], updated[4 * xx + 1], updated[4 * xx + 2] };
                    float alpha = updated[4 * xx + 3];

                    if (normalizedHistogram_)
                    {
                        float binf = x / (float)(canvasSize_.x - 1);
                        binf *= zoomMax_ - zoomMin_;
                        binf += zoomMin_;
                        binf *= normalizedHistogram_->getNumBins() - 1;
                        int bin = (int)binf;

                        int yy = canvasSize_.y - y - 1;
                        if (yy <= (int)normalizedHistogram_->getBinCounts()[bin])
                        {
                            float lum = .3f * rgb.x + .59f * rgb.y + .11f * rgb.z;

                            rgb = { 1.f - lum, 1.f - lum, 1.f - lum };
                        }
                    }

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

        glBindTexture(GL_TEXTURE_2D, prevTexture);

        resetExecutionPolicy();
    }

    void TransfuncEditor::normalizeHistogram()
    {
        assert(vkt::GetThreadExecutionPolicy().device == vkt::ExecutionPolicy::Device::CPU);
        assert(userHistogram_ != nullptr);

        normalizedHistogram_ = new Histogram(userHistogram_->getNumBins());

        std::size_t maxBinCount = 0;

        for (std::size_t i = 0; i < userHistogram_->getNumBins(); ++i)
        {
            maxBinCount = std::max(maxBinCount, userHistogram_->getBinCounts()[i]);
        }

        if (maxBinCount == 0)
        {
            std::fill(
                normalizedHistogram_->getBinCounts(),
                normalizedHistogram_->getBinCounts() + normalizedHistogram_->getNumBins(),
                (size_t)0
                );
        }
        else
        {
            for (std::size_t i = 0; i < userHistogram_->getNumBins(); ++i)
            {
                float countf = true
                    ? logf((float)userHistogram_->getBinCounts()[i]) / logf((float)maxBinCount)
                    : userHistogram_->getBinCounts()[i] / (float)maxBinCount;

                normalizedHistogram_->getBinCounts()[i] = (size_t)(countf * canvasSize_.y);
            }
        }

        histogramChanged_ = false;
    }

    TransfuncEditor::MouseEvent TransfuncEditor::generateMouseEvent()
    {
        setExecutionPolicyCPU();

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

        resetExecutionPolicy();

        return event;
    }

    void TransfuncEditor::handleMouseEvent(TransfuncEditor::MouseEvent const& event)
    {
        setExecutionPolicyCPU();

        bool hovered = ImGui::IsItemHovered()
                && event.pos.x >= 0 && event.pos.x < canvasSize_.x
                && event.pos.y >= 0 && event.pos.y < canvasSize_.y;

        if (event.type == MouseEvent::PassiveMotion || event.type == MouseEvent::Release)
            drawing_ = false;

        if (drawing_ || (event.type == MouseEvent::Press && hovered && event.button == MouseEvent::Left))
        {
            float* updated = (float*)rgbaLookupTable_->getData();

            Vec3i actualDims = rgbaLookupTable_->getDims();

            // Allow for drawing even when we're slightly outside
            // (i.e. not hovering) the drawing area
            int thisX = clamp(event.pos.x, 0, canvasSize_.x - 1);
            int thisY = clamp(event.pos.y, 0, canvasSize_.y - 1);
            int lastX = clamp(lastEvent_.pos.x, 0, canvasSize_.x - 1);

            auto zoom = [=](int x) {
                float indexf = x / (float)(canvasSize_.x - 1);
                indexf *= zoomMax_ - zoomMin_;
                indexf += zoomMin_;
                indexf *= actualDims.x - 1;
                return (int)indexf;
            };

            updated[4 * zoom(thisX) + 3] = thisY / (float)(canvasSize_.y - 1);

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
                    alpha1 = updated[4 * zoom(lastX) + 3];
                    alpha2 = updated[4 * zoom(thisX) + 3];
                }
                else
                {
                    alpha1 = updated[4 * zoom(thisX) + 3];
                    alpha2 = updated[4 * zoom(lastX) + 3];
                }

                int inc = lastEvent_.pos.x < event.pos.x ? 1 : -1;

                for (int x = zoom(lastX) + inc; x != zoom(thisX); x += inc)
                {
                    float frac = (zoom(thisX) - x) / (float)std::abs(zoom(thisX) - zoom(lastX));

                    updated[4 * x + 3] = lerp(alpha1, alpha2, frac);
                }
            }

            lutChanged_ = true;
            drawing_ = true;
        }

        lastEvent_ = event;

        resetExecutionPolicy();
    }

    void TransfuncEditor::setExecutionPolicyCPU()
    {
        vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
        prevPolicy_ = ep;
        ep.device = vkt::ExecutionPolicy::Device::CPU;
        vkt::SetThreadExecutionPolicy(ep);
    }

    void TransfuncEditor::resetExecutionPolicy()
    {
        vkt::SetThreadExecutionPolicy(prevPolicy_);
    }

} // vkt
