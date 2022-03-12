# Toybrot

## TL;DR:

This is some simple code that generates some fractal and I use it to compare
different parallelization models and backends. It optionally uses SDL2 for display and
libPNG for saving the output as images. The build is done with CMake, and tries
to find dependencies as automagically as possible and enable projects and features

## What is ToyBrot?

Toybrot started out initially as a school project for "something that deals
with maths". It was a simple piece of code that calculated a mandelbrot fractal
and drew it to an SDL Surface which then gets drawn on the screen.

Years later, when I was getting to grips with C++11's multithreading support,
I decided to dig this project back up and use it as a demo to get me to implement
some pseudo-real code.

Afterwards I expanded the idea and after some polishing and rework, turned this
project into an experimental ground for my studies of parallelisation in general,
adding different implementations as I got to grips with different programming
models and languages.

Eventually I also decided to start writing a blog series which can be found in
my blog [The Great Refactoring](http://vilelasagna.ddns.net), in a series I called
[Multi Your Threading](http://vilelasagna.ddns.net/tgr/coding/so-you-wanted-to-multi-your-threading-huh/).

As I implement different versions of this code, I write articles on my experiences
with them and compare them to the other implementations, not only in terms of
performance, but also in terms of how difficult I find to actually learn the new
ideas and paradigms and how to put them in code and figure things out when they
don't go quite as expected.

### The Radeon VII incident and the migration to Raymarching

In 2019 I got myself a Radeon VII and it threw somewhat of a wrench in this project
in a curious way. It turns out the Radeon VII is way too powerful for something as
simple as I was doing (the OpenCL implementation averaging about 40ms to generate).

This makes the project very unsuitable for performance comparisons since at that
point any differences will be way too small to be measurable and the set up and
tear down of the subsystems might account for most of the runtime. The answer I
found for this was migrating to a volumetric fractal isntead, using raymarching.
Though there are several implementations designed to run at 60fps on GPUs, it does
provide me the ability to almsot arbitrarily increase the runtime by tweaking the
generation parameters and deliberately choose algorithms such that I can get more
meaningful numbers for comparison


## Building Toybrot

Toybrotis built with C++11 and forward in mind. This is the only consistent
requirement for building it. Other than that it will try and find dependencies
and enable projects as it does through CMake.

The two main dependencies are `SDL2` and `libPNG`. If they are found they enable
a graphical display and the option to save the image to disk when running for
all built projects. Other dependencies will enable more projects when available

Additionally, each built project will come in two flavours. The "regular",
undecorated binary does all the calculations using floats. Binaries appended with
`-db` use doubles throughout. This is achieved by a mix of templated code and
preprocessor shenanigans. Running the project and comparing output should lead
to the conclusion that there's really no reason for doubles other than benchmarking
curiosity.

In order to build the legacy Mandelbrot projects, please enable the CMake option
`ENABLE2D`.

## Running Toybrot

Run the program with -h from the CLI to get more information on the available
options, including benchmarking tweaks

Some common features of note are the already mentioned optional saving of images
as well as the config file. When any of the projects is run, it'll look for a
config file inside `./toybrot.conf.d`. That file has all the tweakable parameters
for controlling the applications function. These include Camera position and
target (though not rotation), parameters for colouring the fractal and for
tweaking the generation of the mandelbox fractal. You can tweak these values and
experiment with different combinations. By default, each implementation has its
own different colours but all other defaults are the same for benchmarking purposes.
To reset any particular setting, you can just delete the line and to reset all,
just delete the file. The missing configurations will get regenerated when the
program next runs.

> All files inside the configuration directory are named `<flavour>.c`the extension
> is just meant to trick various editors into automatically applying some syntax
> highlighting

> When configuring the project, CMake will try and find `std::filesystem` or
> `std::experimental::filesystem`. if it does, the config directory, is created
> automatically if it's not found and, thus, can be deleted to quickly reset all
> variants to default. If the directory is not found, CMake will create it when
> deploying but ToyBrot itself will be unable to re-create it

## Implementations (which I'll try and keep up to date)

 - C++ STL `std::threads`
 - C++ STL `std::async`
 - OpenMP
 - MPI (legacy only)
 - OpenCL
 - nVidia CUDA
 - AMD HC C++ (Legacy only since it's been discontinued)
 - AMD HIP
 - Vulkan
 - SYCL (The implementation I use is [hipSYCL](https://github.com/illuhad/hipSYCL))
 - *(Implementations below this will not be ported to the 2d Mandelbrot version)*
  - Intel TBB
  - Intel ISPC (with `std::async`)
  - Intel ISPC+TBB

## Notes on specific implementations

### CUDA

If CMake detects your `CMKE_CXX_COMPILER` is a new enough clang it'll try
and enable another pair of CUDA projects, so you have `rmCUDA-nvcc[-db]` and
`rmCUDA-clang[-db]`. This can be explicitly disabled through the `BUILD_CUDA_CLANG`
option. If you have a CUDA too new for your clang, but you DO have an older
one as well (which, for example, is my case in Arch Linux), you can point
clang towards the adequate toolkit through the `CUDA_CLANG_DIR` cache variable.
Additionally, if your std c++ headers are too new for NVCC, you can also point
it to an alternative through the `NVCC_CXX_INCLUDES_DIR` cache variable.

### HIP

Though at one point I DID manually trick an NVCC binary into existence that took,
at the time, way more work than I expected, so at this point, it doesn't get
built, just straight up AMD.

### ISPC

ISPC, rather than being a multithreading technology, parallelises your code
inside each thread. So it needs some support to spread it over cores. The
regular ISPC project uses `std::async` to do it, but if CMake also found TBB,
there is another project that pairs both of them. There's no super special
reason for these choices. `std::async` is my default cpu-based option and TBB
gets used a lot in conjunction, them both being from Intel and all, such as in
[OSPRay](https://www.ospray.org/) which uses those two and MPI to spread throug
clusters.

ISPC targets particular instruction set extensions at particular configurations.
Please see [their docs](https://ispc.github.io/ispc.html#selecting-the-compilation-target)
for more info. Toybrot will not normally cross-compile, instead just using
what ISPC detects as the host default (on compile time). If you want to play
around with different targets (either out of curiosity or to examine a specific
system's performance), the option `ISPC_MULTI_TARGET`, when enabled will cause
CMake to build two targets (floats and doubles) for every non-ARM target ISPC
supports (as of version 1.12.0 since I copied the list over to the CMakeLists).
I talk about this in
[my blog](https://vilelasagna.ddns.net/tgr/coding/multi-your-threads-7-feeling-a-little-blue/#Appendix_0x1_Know_your_target).
There's an explanation of how this goes as well as comparative charts for the
performance of the code with different targets both on my 1920X as well as on
an old Haswell-E which can give you some ide of how this variance looks.

### SYCL

As mentioned before the SYCL implementation I use is Illuhad's hipSYCL. This
is built on top of HIP as the name implies. Other implementations might require
massaging to get going.

>Currently SYCL is disabled as it's broken on my machine (it's been a bit janky
>on account of ROCM being all over the place). It's also not got the update with
>the float/double build as well as the config file so it won't build anyway until
>I can get that sorted

### MPI

This was implemented on the legacy project when I had access to a test cluster.
Since that is no longer the case I ended up never going back to it so there is
no MPI implementation for the new raymarching version right now

### Vulkan

Either Vulkan project can choose to load a shader that was compiled from either
a GLSL source or an HLSL one. CMake will look for `glslangValidator` in order to
compile these and will not enable Vulkan if it can't find it. Additionally it'll
also look for [clspv](https://github.com/google/clspv) and, if it finds it, will
use it to compile the same OpenCL code used by the regular OpenCL implementation
into a SPIR shader that can be loaded by Vulkan. The sources themselves are copied
to the deployment directory but this is just illustrative. The binaries for the
shader code are NOT compiled on the fly though you could tweak the source files
there and just call the shader compiler to test new stuff rather than rebuilding
the entire project.

Of note, if you have multiple Vulkan implementations available on your system
you can choose the one you want. `./rmVulkan -l` will list what's found and
then you can use `-v` to choose the one you want specifically. Useful if you
have cards of different vendors, or, say, Mesa's RADV and AMD's AMDVLK present.

### OpenCL

Similar to Vulkan, you can list and choose between different available
implementations and devices on your system. UNLIKE Vulkan, since this is the
standard for OpenCL, the kernel code IS compiled on the fly. So you can tweak
it and run the software to experiment if you want.
