/*
* Copyright 2009-2012  NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* This software and the information contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and conditions
* of a form of NVIDIA software license agreement.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

/** \mainpage
 * \section Introduction
 * The NVIDIA Tools Extension library is a set of functions that a
 * developer can use to provide additional information to tools.
 * The additional information is used by the tool to improve
 * analysis and visualization of data.
 *
 * The library introduces close to zero overhead if no tool is
 * attached to the application.  The overhead when a tool is
 * attached is specific to the tool.
 */

#ifndef NVTOOLSEXT_META_H_
#define NVTOOLSEXT_META_H_

/* Structs defining parameters for NVTX API functions */

struct NvtxMarkEx       { const nvtxEventAttributes_t* eventAttrib; };
struct NvtxMarkA        { const char* message;                      };
struct NvtxMarkW        { const wchar_t* message;                   };
struct NvtxRangeStartEx { const nvtxEventAttributes_t* eventAttrib; };
struct NvtxRangeStartA  { const char* message;                      };
struct NvtxRangeStartW  { const wchar_t* message;                   };
struct NvtxRangeEnd     { nvtxRangeId_t id;                         };
struct NvtxRangePushEx  { const nvtxEventAttributes_t* eventAttrib; };
struct NvtxRangePushA   { const char* message;                      };
struct NvtxRangePushW   { const wchar_t* message;                   };
/*     NvtxRangePop     - no parameters, params will be NULL. */

/* All other NVTX API functions are for naming resources. 
 * A generic params struct is used for all such functions,
 * passing all resource handles as a uint64_t.
 */
typedef struct NvtxNameResourceA
{
    uint64_t resourceHandle;
    const char* name;
} NvtxNameResourceA;

typedef struct NvtxNameResourceW
{
    uint64_t resourceHandle;
    const wchar_t* name;
} NvtxNameResourceW;

#endif /* NVTOOLSEXT_META_H_ */
