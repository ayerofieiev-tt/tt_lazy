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
    }
    default_options = {
        "shared": False,
        "fPIC": True,
    }

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "include/*", "tests/*"

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

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
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
