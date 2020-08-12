// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iostream>

#include "Logging.hpp"

//-------------------------------------------------------------------------------------------------
// Terminal colors
//

#define VKT_TERMINAL_RED     "\033[0;31m"
#define VKT_TERMINAL_GREEN   "\033[0;32m"
#define VKT_TERMINAL_YELLOW  "\033[1;33m"
#define VKT_TERMINAL_RESET   "\033[0m"
#define VKT_TERMINAL_DEFAULT VKT_TERMINAL_RESET
#define VKT_TERMINAL_BOLD    "\033[1;1m"


namespace vkt
{
    namespace logging
    {
        Stream::Stream(Level level)
            : level_(level)
        {
        }

        Stream::~Stream()
        {
            if (level_ == Level::Info)
            {
                std::cout << VKT_TERMINAL_GREEN << stream_.str() << VKT_TERMINAL_RESET << '\n';
            }
        }
    } // logging
} // vkt
