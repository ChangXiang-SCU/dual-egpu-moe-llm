# Building llama.cpp with Vulkan on Windows without the Vulkan SDK

The LunarG Vulkan SDK installer needs administrator rights. On a machine where you do not have
them, you can still build `ggml-vulkan` — the backend needs exactly four things, all obtainable
without an installer:

| what CMake wants | where to get it without the SDK |
|---|---|
| `Vulkan_INCLUDE_DIR` | `git clone https://github.com/KhronosGroup/Vulkan-Headers` |
| `Vulkan_LIBRARY` (`vulkan-1.lib`) | generate an import library from the `vulkan-1.dll` your GPU driver already installed |
| `Vulkan_GLSLC_EXECUTABLE` | a prebuilt `glslc.exe` from shaderc's continuous builds — **version matters, see below** |
| `SPIRV-Headers_DIR` | `git clone https://github.com/KhronosGroup/SPIRV-Headers` + `cmake --install` |

Tested with Visual Studio 2022 Community (MSVC 14.44) and Ninja, on Windows 11.

## 1. Vulkan headers

```bat
git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers.git
```

## 2. `vulkan-1.lib` from the system DLL

The loader DLL ships with every Vulkan-capable GPU driver. Turn it into an import library:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
dumpbin /exports C:\Windows\System32\vulkan-1.dll > exports.txt
:: turn exports.txt into a .def (script below), then:
lib /nologo /def:vulkan-1.def /out:lib\vulkan-1.lib /machine:x64
```

`mkdef.py`:

```python
lines = open('exports.txt', encoding='utf-8', errors='ignore').read().splitlines()
names, on = [], False
for l in lines:
    if 'ordinal hint' in l:
        on = True
        continue
    if on:
        p = l.split()
        if len(p) >= 4 and p[0].isdigit():
            names.append(p[3])
open('vulkan-1.def', 'w').write('LIBRARY vulkan-1\nEXPORTS\n' + '\n'.join(names) + '\n')
print('exports', len(names))
```

265 exports on a current AMD driver. CMake reports the Vulkan version it infers from the headers,
not from this library.

## 3. `glslc.exe` — pick the right build

Prebuilt shaderc binaries are published as Google Cloud Storage artifacts. The download page links
through a redirect stub:

```
https://storage.googleapis.com/shaderc/badges/build_link_windows_vs2022_amd64_release.html
```

which contains a `<meta http-equiv="refresh">` pointing at the actual `install.zip` (~230 MB). To
list what is available:

```
https://storage.googleapis.com/storage/v1/b/shaderc/o?prefix=artifacts/prod/graphics_shader_compiler/shaderc/windows-vs2022-amd64-release/continuous/&fields=items(name,size)
```

**Do not take the newest build blindly.** glslang commit
[`eaff806`](https://github.com/KhronosGroup/glslang/commit/eaff806) (2026-08-24, PR #4052) changed
`LocalSizeId` to be emitted for any SPIR-V >= 1.2 rather than >= 1.6. llama.cpp's Vulkan shaders are
compiled with `--target-env=vulkan1.2`, where spirv-opt then refuses the result:

```
shaderc: internal error: compilation succeeded but failed to optimize:
LocalSizeId mode is not allowed by the current environment.
This is allowed if you enable the maintenance4 feature (or use the --allow-localsizeid flag)
```

Any shaderc build made after that commit fails on the very first shader (`argmax.comp`). Build **37
(2026-08-06)** works; build 38 (2026-08-25) does not. If you must use a newer glslc, pass
`--allow-localsizeid` or raise the target environment instead.

## 4. SPIRV-Headers

`ggml/src/ggml-vulkan/CMakeLists.txt` does `find_package(SPIRV-Headers CONFIG REQUIRED)`, so
headers on the include path are not enough — it needs the installed CMake config package:

```bat
git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git
cmake -S SPIRV-Headers -B SPIRV-Headers\build -G Ninja ^
      -DCMAKE_INSTALL_PREFIX=%CD%\spv -DSPIRV_HEADERS_ENABLE_TESTS=OFF
cmake --build SPIRV-Headers\build --target install
```

`ggml-vulkan.cpp` then `#include`s `<spirv/unified1/spirv.hpp>` directly. The CMake package sets
`SPIRV-Headers_DIR` but does not add that include path to the `ggml-vulkan` target, so on this
layout you also need the headers reachable from the Vulkan include dir:

```bat
xcopy /E /I /Y spv\include\spirv Vulkan-Headers\include\spirv
```

## 5. Configure and build

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;^
C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_VULKAN=ON ^
  -DVulkan_INCLUDE_DIR=%VK%\Vulkan-Headers\include ^
  -DVulkan_LIBRARY=%VK%\lib\vulkan-1.lib ^
  -DVulkan_GLSLC_EXECUTABLE=%VK%\install\bin\glslc.exe ^
  -DSPIRV-Headers_DIR=%VK%\spv\share\cmake\SPIRV-Headers ^
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DBUILD_SHARED_LIBS=ON ^
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --config Release -j 8
```

`GGML_BACKEND_DL=ON` + `GGML_CPU_ALL_VARIANTS=ON` + `BUILD_SHARED_LIBS=ON` reproduces the layout of
the official Windows releases (`llama-server.exe` plus `ggml-vulkan.dll` and a set of
`ggml-cpu-*.dll`), so the result drops in beside an existing install. About 15 minutes on a
Core Ultra 9 185H.

## Deploying to another Windows box

Copy `build\bin\*.exe` and `build\bin\*.dll`. There is no Vulkan runtime to install — `vulkan-1.dll`
is already there as part of the GPU driver, and it is the same DLL the import library was generated
from.

## Useful environment variables when something looks wrong

- `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` — turns off Vulkan graph reordering and operator fusion. Worth
  trying first whenever a graph modification produces wrong results; the topk-moe fusion matches its
  pattern by fixed node offsets and is easy to disturb.
- `LLAMA_GRAPH_REUSE_DISABLE=1` — rebuilds the compute graph every call instead of reusing it.
- `GGML_VK_DISABLE_COOPMAT=1`, `GGML_VK_DISABLE_F16=1` — narrow down numerical differences.
