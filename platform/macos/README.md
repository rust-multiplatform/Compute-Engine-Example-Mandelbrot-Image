# Platform: macOS

[![Rust](https://github.com/rust-multiplatform/Compute-Engine-Example-Mandelbrot-Image/actions/workflows/platform_macos.yml/badge.svg)](https://github.com/rust-multiplatform/Compute-Engine-Example-Mandelbrot-Image/actions/workflows/platform_macos.yml)
[![codecov](https://codecov.io/gh/rust-multiplatform/Compute-Engine-Example-Mandelbrot-Image/branch/main/graph/badge.svg?token=XpGvuQVirP)](https://codecov.io/gh/rust-multiplatform/Compute-Engine-Example-Mandelbrot-Image)

This is the platform-specific project for the platform macOS.  
Make sure to check the shared crate as we depend on it.

To get any logging outputs run this with the environment variable `RUST_LOG` set to `INFO` or `DEBUG`.
Alternatively, running this in debug mode _should_ automatically log everything in `DEBUG` mode.

## How does this work?

Rust is a cross-platform language that supports many different architectures and platforms.  
Building for Desktop Platforms (like Windows, Linux, macOS) is natively supported as long as the required tools are installed.
For Android, iOS and WebAssembly (WASM) we _"require"_ some additional tools to build.
In essence, we could do it without those tools but it would be much harder to do so.
Said tools commonly package our apps into .APK, .APP and .JS/WASM, link them against our library and make sure the entrypoint is called.

Inside `src/lib.rs` we have a public entrypoint defined:

```rust
pub fn entrypoint() {
    // Our cross-platform code here
}
```

Each platform calls this, while each platform itself has an entrypoint.  

Ideally, we **only** have our shared code.  
However, sometimes we need some very platform specific things that can only be done on said platform.
In this case we have _some_ non-shared/platform-specific code but overall this should only be true for very special cases as the Rust standard library works mostly the same on each platform/target.

## What does this crate do?

We implement the standard Rust Binary-Entrypoint:

```rust
fn main() {
    // OUR CODE HERE
}
```

And simply call the entrypoint defined by our shared library.  
Check out the shared crate for more on how this is working and what it is doing.

This is the most basic of examples: A simple `"Hello World"` or better yet: `"Hello from Rust!"` to the standard output.  
On Desktop platforms this will open a Terminal/Console printing out the text and exiting.  
For Android, iOS and WebAssembly this will be logged to LogCat, Device Logs and the Browser Console respectively.  

**Independent of the platform the app will _exit_ right after being done. Depending on the platform the window (console _or_ app) may close before you even see it.**  
On Desktop you can call the executable from a terminal either by using the executable in `target/(debug|release|<arch>)/platform_*(.exe)` or using the `cargo run --package platform_<platform>` command.
For Android, iOS and WebAssembly enable persistent logging and reopen the app/website.

## Building

|                         | Host: Windows                                                                                                                                                                                                                                                    | Host: Linux                                                                                                                                                                     | Host: macOS                                                                                                                                    |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Target: Windows**     | ✅: [Visual Studio](https://visualstudio.com/)                                                                                                                                                                                                                    | 🔀: [MinGW](https://www.mingw-w64.org/)                                                                                                                                          | 🔀: [MinGW](https://www.mingw-w64.org/)                                                                                                         |
| **Target: Linux**       | ⚠️: [WSL](https://docs.microsoft.com/en-us/windows/wsl/) or VM or Docker                                                                                                                                                                                          | ✅: [GCC](https://gcc.gnu.org/) or [Clang](https://clang.llvm.org/)                                                                                                              | 🔀: Docker or VM                                                                                                                                |
| **Target: macOS**       | ⚠️: [Docker-OSX (inside WSL with Docker)](https://github.com/sickcodes/Docker-OSX) or [OSX-KVM (inside WSL or VM)](https://github.com/kholia/OSX-KVM) or [macOS-VirtualBox (inside/with WSL and/or MSYS2/MinGW)](https://github.com/myspaghetti/macos-virtualbox) | ⚠️: [Docker-OSX](https://github.com/sickcodes/Docker-OSX) or [OSX-KVM](https://github.com/kholia/OSX-KVM) or [macOS-VirtualBox](https://github.com/myspaghetti/macos-virtualbox) | ✅: [XCode](https://developer.apple.com/xcode/)                                                                                                 |
| **Target: Android**     | 🔀: [Android Studio](https://developer.android.com/studio/) or [Android CommandLine-Tools](https://developer.android.com/studio/#command-tools)                                                                                                                   | 🔀: [Android Studio](https://developer.android.com/studio/) or [Android CommandLine-Tools](https://developer.android.com/studio/#command-tools)                                  | 🔀: [Android Studio](https://developer.android.com/studio/) or [Android CommandLine-Tools](https://developer.android.com/studio/#command-tools) |
| **Target: iOS**         | ⚠️: [Docker-OSX (inside WSL with Docker)](https://github.com/sickcodes/Docker-OSX) or [OSX-KVM (inside WSL or VM)](https://github.com/kholia/OSX-KVM) or [macOS-VirtualBox (inside/with WSL and/or MSYS2/MinGW)](https://github.com/myspaghetti/macos-virtualbox) | ⚠️: [Docker-OSX](https://github.com/sickcodes/Docker-OSX) or [OSX-KVM](https://github.com/kholia/OSX-KVM) or [macOS-VirtualBox](https://github.com/myspaghetti/macos-virtualbox) | ✅: [XCode](https://developer.apple.com/xcode/)                                                                                                 |
| **Target: WebAssembly** | ✅: [Wasm-Pack](https://rustwasm.github.io/wasm-pack/installer/)                                                                                                                                                                                                  | ✅: [Wasm-Pack](https://rustwasm.github.io/wasm-pack/installer/)                                                                                                                 | ✅: [Wasm-Pack](https://rustwasm.github.io/wasm-pack/installer/)                                                                                |

✅ = Natively supported.  
🔀 = Cross-Compilation & Toolchain needed.  
⚠️ = Possible, but takes some more effort and/or special setups or VM to work.

Requirements:  

- Rust and Cargo is installed, preferably via [Rustup](https://rustup.rs/). Use the stable channel ideally.
- Rust on macOS requires Clang to be installed, ideally install XCode from the AppStore (or archive for older macOS versions) or use [Brew](https://brew.sh) to install `xcode-buildtools` and `clang`.

Debug/Development Build:

```shell
cargo build --package platform_macos
```

Release/Production Build:

```shell
cargo build --package platform_macos --release
```

> Optionally add `--verbose` to see what is happening.  
> Floods your console with message though possibly, use when there are build errors.

## Testing

Requirements:  

- Rust and Cargo is installed, preferably via [Rustup](https://rustup.rs/). Use the stable channel ideally.
- Rust on macOS requires Clang to be installed, ideally install XCode from the AppStore (or archive for older macOS versions) or use [Brew](https://brew.sh) to install `xcode-buildtools` and `clang`.

Debug/Development Testing:

```shell
cargo test --package platform_macos
```

Release/Production Testing:

```shell
cargo test --package platform_macos --release
```

> Optionally add `--verbose` to see what is happening.  
> Floods your console with message though possibly, use when there are build errors.

## Running

Requirements:  

- Rust and Cargo is installed, preferably via [Rustup](https://rustup.rs/). Use the stable channel ideally.
- Rust on macOS requires Clang to be installed, ideally install XCode from the AppStore (or archive for older macOS versions) or use [Brew](https://brew.sh) to install `xcode-buildtools` and `clang`.

Debug/Development Running:

```shell
cargo run --package platform_macos
```

Release/Production Running:

```shell
cargo run --package platform_macos --release
```

> Optionally add `--verbose` to see what is happening.  
> Floods your console with message though possibly, use when there are build errors.
