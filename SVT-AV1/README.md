# Scalable Video Technology for AV1 (SVT-AV1 Encoder and Decoder)

The Scalable Video Technology for AV1 (SVT-AV1 Encoder and Decoder) is an AV1-compliant encoder/decoder library core. The SVT-AV1 encoder development is a work-in-progress targeting performance levels applicable to both VOD and Live encoding / transcoding video applications. The SVT-AV1 decoder implementation is targeting future codec research activities.

## License

Up to v0.8.7, SVT-AV1 is licensed under the BSD-2-clause license and the Alliance for Open Media Patent License 1.0. See [LICENSE](LICENSE-BSD2.md) and [PATENTS](PATENTS.md) for details.
Starting from v0.8.8, SVT-AV1 is licensed under the BSD-3-clause clear license and the Alliance for Open Media Patent License 1.0. See [LICENSE](LICENSE.md) and [PATENTS](PATENTS.md) for details.

## Documentation

More details about the SVT-AV1 usage and implementation can be found under:

- [svt-av1-encoder-user-guide](Docs/svt-av1_encoder_user_guide.md)
- [svt-av1-decoder-user-guide](Docs/svt-av1_decoder_user_guide.md)
- [svt-av1-documentation-page](Docs/README.md)

## System Requirements

### Operating System

SVT-AV1 Encoder may run on any Windows* or Linux* 64 bit operating systems. The list below represents the operating systems that the encoder application and library were tested and validated on:

- __Windows* Operating Systems (64-bit):__
  - Windows* Server 2016
- __Linux* Operating Systems (64-bit):__
  - Ubuntu* 16.04 Server LTS
  - Ubuntu* 18.04 Server LTS
- __Unix* Operating Systems (64-bit):__
  - MacOS

### Hardware

The SVT-AV1 Encoder library supports the x86 architecture

- __CPU Requirements__

    In order to achieve the performance targeted by the SVT-AV1 Encoder, the specific CPU model listed above would need to be used when running the encoder. Otherwise, the encoder runs on any 5th Generation Intel® Core™ processor, (Intel® Xeon® CPUs, E5-v4 or newer).

- __RAM Requirements__

    The SVT-AV1 Encoder adapts to the system on which it is being run. The memory requirements depend on the number of cores the system contains, the input frame rate of the input sequence (`-fps`) and the look ahead distance passed to the encoder (`-lad`). The SVT-AV1 Encoder application will display an error if the system does not have enough RAM to support the encode prior to the start of the encode. The following table shows the minimum amount of RAM required for some standard resolutions of 10bit video per stream:

|       Resolution      | 8-vCPU Commit Size (GB)| 40-vCPU Commit Size (GB)|
|-----------------------|------------------------|-------------------------|
|       4k              |           14           |           24            |
|       1080p           |            6           |           10            |
|       720p            |            4           |            7            |
|       480p            |            3           |            5            |

## Build and Install

### Windows* Operating Systems (64-bit)

- __Build Requirements__
  - Visual Studio* 2017 (download [here](https://www.visualstudio.com/vs/older-downloads/)) or 2019 (download [here](https://visualstudio.microsoft.com/downloads/))
  - CMake 3.5 or later (download [here](https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5-win64-x64.msi))
  - YASM Assembler version 1.2.0 or later
    - Download the yasm exe from the following [link](http://www.tortall.net/projects/yasm/releases/yasm-1.3.0-win64.exe)
    - Rename yasm-*-win64.exe to yasm.exe
    - Copy yasm.exe into a location that is in the `PATH` environment variable

- __Build Instructions__
  - Build the project by following the steps below
    - cd into `Build\windows`
    - run `build.bat <2019|2017|2015>` [This will generate the .sln files and build the project]

- __Binaries Location__
  - Binaries can be found under `<repo dir>/Bin/Release` or `<repo dir>/Bin/Debug`, depending on whether Debug or Release were selected in the build mode.

- __Installation__

  For the binaries to operate properly on your system, the following conditions have to be met:
  - On any of the Windows* Operating Systems listed in the OS requirements section, install Visual Studio* 2015/2017/2019
  - Once the installation is complete, copy the binaries to a location making sure that both the sample application `SvtAv1EncApp.exe` and library `SvtAv1Enc.dll` are in the same folder.
  - Open the command prompt window at the chosen location and run the sample application to encode: `SvtAV1EncApp.exe -i [in.yuv] -w [width] -h [height] -b [out.ivf]`
  - Sample application supports reading from pipe. E.g. `ffmpeg -i [input.mp4] -nostdin -f rawvideo -pix_fmt yuv420p - | SvtAv1EncApp.exe -i stdin -n [number_of_frames_to_encode] -w [width] -h [height]`

### Linux* Operating Systems (64-bit)

Note - a Dockerfile is provided to build the encoder into a tiny Alpine Linux Docker image.

- __Build Requirements__
  - GCC 5.4.0 or later
  - CMake 3.5.1 or later
  - YASM Assembler version 1.2.0 or later

- __Build Instructions__
  - `cd Build/linux`
  - `./build.sh <release | debug>`

- __Sample Binaries location__
  - Binaries can be found under `Bin/Release` and/or `Bin/Debug`

- __Clang usage__
  - To install Clang-11 on Ubuntu 20.04 execute single command: `sudo apt install clang-11`
  - To install Clang-11 on Ubuntu 18.04 execute commands:
    - `wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -`
    - `sudo apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main"`
    - `sudo apt install clang-11`
  - To build SVT-AV1 using Clang-11:
    - `export CC="clang-11"`
    - `export CXX="clang++-11"`
    - `./build.sh <release | debug>`


- __Installation__

  For the binaries to operate properly on your system, the following conditions have to be met:

  - On any of the Linux* Operating Systems listed above, copy the binaries under a location of your choice.
  - Change the permissions on the sample application `SvtAV1EncApp` executable by running the command: `chmod +x SvtAv1EncApp`
  - cd into your chosen location
  - Run the sample application to encode: `./SvtAv1EncApp -i [in.yuv] -w [width] -h [height] -b [out.ivf]`
  - Sample application supports reading from pipe. E.g. `ffmpeg -i [input.mp4] -nostdin -f rawvideo -pix_fmt yuv420p - | ./SvtAv1EncApp -i stdin -n [number_of_frames_to_encode] -w [width] -h [height]`

## SVT-AV1 ffmpeg plugin installation

### 1. Build and install SVT-AV1

``` bash
git clone --depth=1 https://gitlab.com/AOMediaCodec/SVT-AV1.git
cd SVT-AV1
cd Build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make -j $(nproc)
sudo make install
```

###  2. Enable libsvtav1 in FFmpeg

NOTE: If you wish to use an FFmpeg tag or release before 4.4, please go [here](https://gitlab.com/AOMediaCodec/SVT-AV1/tree/v0.8.4/ffmpeg_plugin) and consult that page to properly patch ffmpeg for use with SVT-AV1.

``` bash
   git clone --depth=1 https://github.com/FFmpeg/FFmpeg ffmpeg
   cd ffmpeg
   export LD_LIBRARY_PATH+=":/usr/local/lib"
   export PKG_CONFIG_PATH+=":/usr/local/lib/pkgconfig"
   ./configure --enable-libsvtav1
   make -j $(nproc)
```

###  3. Verify that ffmpeg is working

``` bash
./ffmpeg -i input.mp4 -c:v libsvtav1 -y test.mp4
```

## How to evaluate by ready-to-run executables with docker

Refer to the guide [here](https://github.com/OpenVisualCloud/Dockerfiles/blob/master/doc/svt.md#Evaluate-SVT).

## Demo features and limitations

- **Multi-instance support:** The multi-instance functionality is a demo feature implemented in the SVT-AV1 Encoder sample application as an example of one sample application using multiple encoding libraries. Encoding using the multi-instance support is limited to only 6 simultaneous streams. For example two channels encoding on Windows: `SvtAV1EncApp.exe -nch 2 -c firstchannel.cfg secondchannel.cfg`
- **Features enabled:** The library will display an error message any feature combination that is not currently supported.

## How to Contribute

We welcome community contributions to the SVT-AV1 Encoder and Decoder. Thank you for your time! By contributing to the project, you agree to the license, patent and copyright terms in the AOM License and Patent License  and to the release of your contribution under these terms. See [LICENSE](LICENSE.md) and [PATENTS](PATENTS.md) for details.

## Contributor agreement

You will be required to execute the appropriate [contributor agreement](http://aomedia.org/license/) to ensure that the AOMedia Project has the right to distribute your changes.

### Contribution process

- Follow the [coding guidelines](STYLE.md) and the [contributing guidelines](CONTRIBUTING.md)

- Validate that your changes do not break a build

- Perform smoke tests and ensure they pass

- Submit a pull request for review to the maintainer

### How to Report Bugs and Provide Feedback

Use the [Issues](https://gitlab.com/AOMediaCodec/SVT-AV1/issues) tab on GitLab. To avoid duplicate issues, please make sure you go through the existing issues before logging a new one.

## IRC

`#svt` on Libera.chat. Join via [Libera Webchat](https://web.libera.chat/?channel=#svt) or use your favorite IRC client.

## Performance tracker

Performance and raw data files available at <http://videocodectracker.dev>
