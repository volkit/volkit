// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/config.h>

#if VKT_HAVE_GLEW
#include <GL/glew.h>
#endif

#include <cstdint>

#include <vkt/common.hpp>

namespace vkt
{
    struct ColorFormatInfo
    {
        ColorFormat colorFormat;
        uint8_t components;
        uint8_t size;
    };

    static ColorFormatInfo ColorFormatInfoTable[(int)ColorFormat::Count] = {
            { ColorFormat::Unspecified,     0,   0      },
            { ColorFormat::R8,              1,   1      },
            { ColorFormat::RG8,             2,   2      },
            { ColorFormat::RGB8,            3,   3      },
            { ColorFormat::RGBA8,           4,   4      },
            { ColorFormat::R16UI,           1,   2      },
            { ColorFormat::RG16UI,          2,   4      },
            { ColorFormat::RGB16UI,         3,   6      },
            { ColorFormat::RGBA16UI,        4,   8      },
            { ColorFormat::R32UI,           1,   4      },
            { ColorFormat::RG32UI,          2,   8      },
            { ColorFormat::RGB32UI,         3,  12      },
            { ColorFormat::RGBA32UI,        4,  16      },
            { ColorFormat::R32F,            1,   4      },
            { ColorFormat::RG32F,           2,   8      },
            { ColorFormat::RGB32F,          3,  12      },
            { ColorFormat::RGBA32F,         4,  16      },

    };

#if VKT_HAVE_GLEW
    struct ColorFormatInfoGL
    {
        ColorFormat colorFormat;
        GLuint format;
        GLuint internalFormat;
        GLuint type;
    };

    static ColorFormatInfoGL ColorFormatInfoTableGL[(int)ColorFormat::Count] = {
            { ColorFormat::Unspecified, 0,               0,           0                },
            { ColorFormat::R8,          GL_RED,          GL_R8,       GL_UNSIGNED_BYTE },
            { ColorFormat::RG8,         GL_RG,           GL_RG8,      GL_UNSIGNED_BYTE },
            { ColorFormat::RGB8,        GL_RGB,          GL_RGB8,     GL_UNSIGNED_BYTE },
            { ColorFormat::RGBA8,       GL_RGBA,         GL_RGBA8,    GL_UNSIGNED_BYTE },
            { ColorFormat::R16UI,       GL_RED_INTEGER,  GL_R16UI,    GL_UNSIGNED_INT  },
            { ColorFormat::RG16UI,      GL_RG_INTEGER,   GL_RG16UI,   GL_UNSIGNED_INT  },
            { ColorFormat::RGB16UI,     GL_RGB_INTEGER,  GL_RGB16UI,  GL_UNSIGNED_INT  },
            { ColorFormat::RGBA16UI,    GL_RGBA_INTEGER, GL_RGBA16UI, GL_UNSIGNED_INT  },
            { ColorFormat::R32UI,       GL_RED_INTEGER,  GL_R32UI,    GL_UNSIGNED_INT  },
            { ColorFormat::RG32UI,      GL_RG_INTEGER,   GL_RG32UI,   GL_UNSIGNED_INT  },
            { ColorFormat::RGB32UI,     GL_RGB_INTEGER,  GL_RGB32UI,  GL_UNSIGNED_INT  },
            { ColorFormat::RGBA32UI,    GL_RGBA_INTEGER, GL_RGBA32UI, GL_UNSIGNED_INT  },
            { ColorFormat::R32F,        GL_RED,          GL_R32F,     GL_FLOAT         },
            { ColorFormat::RG32F,       GL_RG,           GL_RG32F,    GL_FLOAT         },
            { ColorFormat::RGB32F,      GL_RGB,          GL_RGB32F,   GL_FLOAT         },
            { ColorFormat::RGBA32F,     GL_RGBA,         GL_RGBA32F,  GL_FLOAT         },

    };

#endif
} // vkt
