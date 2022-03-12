/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbApiVersion_h
#define EbApiVersion_h

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// API Version
#define SVT_VERSION_MAJOR       (0)
#define SVT_VERSION_MINOR       (3)
#define SVT_VERSION_PATCHLEVEL  (0)

#define    SVT_CHECK_VERSION(major,minor,patch)    \
    (SVT_VERSION_MAJOR > (major) || \
     (SVT_VERSION_MAJOR == (major) && SVT_VERSION_MINOR > (minor)) || \
     (SVT_VERSION_MAJOR == (major) && SVT_VERSION_MINOR == (minor) && \
      SVT_VERSION_PATCHLEVEL >= (patch)))

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // EbApiVersion_h
