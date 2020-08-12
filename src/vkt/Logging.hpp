// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <ostream>
#include <sstream>

namespace vkt
{
    namespace logging
    {
        enum class Level
        {
            Error,
            Warning,
            Info,
        };

        class Stream
        {   

        public:
            Stream(Level level);
           ~Stream();

            inline std::ostream& stream()
            {
                return stream_;
            }

        private:
            std::ostringstream stream_;
            Level level_;

        };
    } // logging
} // vkt


#define VKT_LOG(LEVEL) ::vkt::logging::Stream(LEVEL).stream()
