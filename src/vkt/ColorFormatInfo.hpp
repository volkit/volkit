#pragma once

#include <cstdint>

#include <vkt/common.hpp>

namespace vkt
{
    struct ColorFormatInfo
    {
        ColorFormat format;
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
} // vkt
