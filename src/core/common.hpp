#pragma once
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

template <typename T, size_t N = 4>
using SmallVector = boost::container::small_vector<T, N>;

// Logger configuration
namespace tt_lazy {
    inline void setup_logging() {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");
        
        auto logger = std::make_shared<spdlog::logger>("tt_lazy", console_sink);
        logger->set_level(spdlog::level::info);
        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger);
    }
    
    inline spdlog::logger& get_logger() {
        return *spdlog::get("tt_lazy");
    }
}
