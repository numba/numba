/* Copyright 2014 HSA Foundation Inc.  All Rights Reserved.
 *
 * HSAF is granting you permission to use this software and documentation (if
 * any) (collectively, the "Materials") pursuant to the terms and conditions
 * of the Software License Agreement included with the Materials.  If you do
 * not have a copy of the Software License Agreement, contact the  HSA Foundation for a copy.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
 */

#pragma once

#include "hsa_ext_finalize.h"

typedef enum status_t status_t;
enum status_t {
    STATUS_SUCCESS=0,
    STATUS_KERNEL_INVALID_SECTION_HEADER=1,
    STATUS_KERNEL_ELF_INITIALIZATION_FAILED=2,
    STATUS_KERNEL_INVALID_ELF_CONTAINER=3,
    STATUS_KERNEL_MISSING_DATA_SECTION=4,
    STATUS_KERNEL_MISSING_CODE_SECTION=5,
    STATUS_KERNEL_MISSING_OPERAND_SECTION=6,
    STATUS_UNKNOWN=7,
};

status_t create_brig_module_from_brig_file(const char* file_name, hsa_ext_brig_module_t** brig_module);

void destroy_brig_module(hsa_ext_brig_module_t* brig_module);
