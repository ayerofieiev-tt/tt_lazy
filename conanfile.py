from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


class TtLazyConan(ConanFile):
    name = "tt_lazy"
    version = "1.0.0"
    package_type = "library"

    # Optional metadata
    license = "MIT"
    author = "Your Name <your.email@example.com>"
    url = "https://github.com/yourusername/tt_lazy"
    description = "High-performance C++ ML framework with lazy evaluation"
    topics = ("machine-learning", "tensor", "lazy-evaluation", "c++")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "enable_asan": [True, False],
        "enable_ubsan": [True, False],
        "enable_clang_tidy": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "enable_asan": False,
        "enable_ubsan": False,
        "enable_clang_tidy": False,
    }

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "core/*", "include/*", "tests/*"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("gtest/1.14.0")
        self.requires("boost/1.84.0")
        self.requires("pybind11/2.12.0")
        self.requires("spdlog/1.12.0")

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)

        # Pass sanitizer and analysis options to CMake
        tc.cache_variables["ENABLE_ASAN"] = self.options.enable_asan
        tc.cache_variables["ENABLE_UBSAN"] = self.options.enable_ubsan
        tc.cache_variables["ENABLE_CLANG_TIDY"] = self.options.enable_clang_tidy

        # Set defaults based on build type
        if self.settings.build_type == "Debug":
            tc.cache_variables["ENABLE_ASAN"] = True
            tc.cache_variables["ENABLE_UBSAN"] = True
            tc.cache_variables["ENABLE_CLANG_TIDY"] = True
        elif self.settings.build_type == "Release":
            tc.cache_variables["ENABLE_ASAN"] = False
            tc.cache_variables["ENABLE_UBSAN"] = False
            tc.cache_variables["ENABLE_CLANG_TIDY"] = False

        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["tt_lazy_lib"]
