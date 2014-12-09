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

#include <stdlib.h>
#include <libelf.h>
#include <string.h>
#include <stdio.h>
#include "hsa.h"
#include "elf_utils.h"
#include "hsa_ext_finalize.h"

enum {
    SECTION_HSA_DATA = 0,
    SECTION_HSA_CODE,
    SECTION_HSA_OPERAND,
};

typedef struct SectionDesc SectionDesc;
struct SectionDesc {
  int sectionId;
  const char *brigName;
  const char *bifName;
}

sectionDescs[] = {
    { SECTION_HSA_DATA, "hsa_data",".brig_hsa_data" },
    { SECTION_HSA_CODE, "hsa_code",".brig_hsa_code" },
    { SECTION_HSA_OPERAND,"hsa_operand",".brig_hsa_operand"},
};

extern int fileno(FILE* stream);

const SectionDesc* get_section_desc(int sectionId) {
    const int NUM_PREDEFINED_SECTIONS = sizeof(sectionDescs)/sizeof(sectionDescs[0]);
    for(int i=0; i<NUM_PREDEFINED_SECTIONS; ++i) {
        if (sectionDescs[i].sectionId == sectionId) {
            return &sectionDescs[i];
        }
    }
    return NULL;
}

static Elf_Scn* extract_elf_section (Elf *elfP,
  				   Elf_Data *secHdr,
				   const SectionDesc* desc) {
    int cnt = 0;
    Elf_Scn* scn = NULL;
    Elf32_Shdr* shdr = NULL;
    char* sectionName = NULL;

    /* Iterate thru the elf sections */
    for (cnt = 1, scn = NULL; scn = elf_nextscn(elfP, scn); cnt++) {
        if (((shdr = elf32_getshdr(scn)) == NULL)) {
            return NULL;
        }
        sectionName = (char *)secHdr->d_buf + shdr->sh_name;
        if (sectionName &&
           ((strcmp(sectionName, desc->brigName) == 0) ||
           (strcmp(sectionName, desc->bifName) == 0))) {
            return scn;
        }
     }

     return NULL;
}

/* Extract section and copy into HsaBrig */
static status_t extract_section_and_copy (Elf *elfP,
                                       Elf_Data *secHdr, 
                                       const SectionDesc* desc,
                                       hsa_ext_brig_module_t* brig_module,
                                       hsa_ext_brig_section_id_t section_id) {
    Elf_Scn* scn = NULL;
    Elf_Data* data = NULL;
    void* address_to_copy;
    size_t section_size=0;

    scn = extract_elf_section(elfP, secHdr, desc);

    if (scn) {
        if ((data = elf_getdata(scn, NULL)) == NULL) {
            return STATUS_UNKNOWN;
        }
        section_size = data->d_size;
        if (section_size > 0) {
          address_to_copy = malloc(section_size);
          memcpy(address_to_copy, data->d_buf, section_size);
        }
    }

    if ((!scn ||  section_size == 0))  {
        return STATUS_UNKNOWN;
    }

    /* Create a section header */
    brig_module->section[section_id] = (hsa_ext_brig_section_header_t*) address_to_copy; 

    return STATUS_SUCCESS;
} 

/* Reads binary of BRIG and BIF format */
status_t read_binary(hsa_ext_brig_module_t **brig_module_t, FILE* binary) {
    /* Create the brig_module */
    uint32_t number_of_sections = 3;
    hsa_ext_brig_module_t* brig_module;

    brig_module = (hsa_ext_brig_module_t*)
                  (malloc (sizeof(hsa_ext_brig_module_t) + sizeof(void*)*number_of_sections));
    brig_module->section_count = number_of_sections;

    status_t status;
    Elf* elfP = NULL;
    Elf32_Ehdr* ehdr = NULL;
    Elf_Data *secHdr = NULL;
    Elf_Scn* scn = NULL;
    int fd;

    if (elf_version ( EV_CURRENT ) == EV_NONE) {
        return STATUS_KERNEL_ELF_INITIALIZATION_FAILED;
    } 

    fd = fileno(binary);
    if ((elfP = elf_begin(fd, ELF_C_READ, (Elf *)0)) == NULL) {
        return STATUS_KERNEL_INVALID_ELF_CONTAINER;
    }

    if (elf_kind (elfP) != ELF_K_ELF) {
        return STATUS_KERNEL_INVALID_ELF_CONTAINER;
    }
  
    if (((ehdr = elf32_getehdr(elfP)) == NULL) ||
       ((scn = elf_getscn(elfP, ehdr->e_shstrndx)) == NULL) ||
       ((secHdr = elf_getdata(scn, NULL)) == NULL)) {
        return STATUS_KERNEL_INVALID_SECTION_HEADER;
    }

    status = extract_section_and_copy(elfP, 
                                   secHdr,
                                   get_section_desc(SECTION_HSA_DATA), 
                                   brig_module,
                                   HSA_EXT_BRIG_SECTION_DATA);

    if (status != STATUS_SUCCESS) {
        return STATUS_KERNEL_MISSING_DATA_SECTION;
    }

    status = extract_section_and_copy(elfP, 
				   secHdr,
                                   get_section_desc(SECTION_HSA_CODE), 
                                   brig_module,
                                   HSA_EXT_BRIG_SECTION_CODE);

    if (status != STATUS_SUCCESS) {
        return STATUS_KERNEL_MISSING_CODE_SECTION;
    }

    status = extract_section_and_copy(elfP, 
                                   secHdr,
                                   get_section_desc(SECTION_HSA_OPERAND), 
                                   brig_module,
                                   HSA_EXT_BRIG_SECTION_OPERAND);

    if (status != STATUS_SUCCESS) {
        return STATUS_KERNEL_MISSING_OPERAND_SECTION;
    }

    elf_end(elfP);
    *brig_module_t = brig_module;

    return STATUS_SUCCESS;
}

status_t create_brig_module_from_brig_file(const char* file_name, hsa_ext_brig_module_t** brig_module) {
    FILE *fp = fopen(file_name, "rb");

    status_t status = read_binary(brig_module, fp);

    if (status != STATUS_SUCCESS) {
        printf("Could not create BRIG module: %d\n", status);
        if (status == STATUS_KERNEL_INVALID_SECTION_HEADER || 
            status == STATUS_KERNEL_ELF_INITIALIZATION_FAILED || 
            status == STATUS_KERNEL_INVALID_ELF_CONTAINER) {
            printf("The ELF file is invalid or possibley corrupted.\n");
        }
        if (status == STATUS_KERNEL_MISSING_DATA_SECTION ||
            status == STATUS_KERNEL_MISSING_CODE_SECTION ||
            status == STATUS_KERNEL_MISSING_OPERAND_SECTION) {
            printf("One or more ELF sections are missing. Use readelf command to \
            to check if hsa_data, hsa_code and hsa_operands exist.\n");
        }
    }

    fclose(fp);

    return status;
}

void destroy_brig_module(hsa_ext_brig_module_t* brig_module) {
    for (int i=0; i<brig_module->section_count; i++) {
        free (brig_module->section[i]);
    }
    free (brig_module);
}


/*
 * Finds the specified symbols offset in the specified brig_module.
 * If the symbol is found the function returns HSA_STATUS_SUCCESS, 
 * otherwise it returns HSA_STATUS_ERROR.
 */
hsa_status_t find_symbol_offset(hsa_ext_brig_module_t* brig_module, 
    char* symbol_name,
    hsa_ext_brig_code_section_offset32_t* offset) {
    
    /* 
     * Get the data section 
     */
    hsa_ext_brig_section_header_t* data_section_header = 
                brig_module->section[HSA_EXT_BRIG_SECTION_DATA];
    /* 
     * Get the code section
     */
    hsa_ext_brig_section_header_t* code_section_header =
             brig_module->section[HSA_EXT_BRIG_SECTION_CODE];

    /* 
     * First entry into the BRIG code section
     */
    BrigCodeOffset32_t code_offset = code_section_header->header_byte_count;
    BrigBase* code_entry = (BrigBase*) ((char*)code_section_header + code_offset);
    while (code_offset != code_section_header->byte_count) {
        if (code_entry->kind == BRIG_KIND_DIRECTIVE_KERNEL) {
            /* 
             * Now find the data in the data section
             */
            BrigDirectiveExecutable* directive_kernel = (BrigDirectiveExecutable*) (code_entry);
            BrigDataOffsetString32_t data_name_offset = directive_kernel->name;
            BrigData* data_entry = (BrigData*)((char*) data_section_header + data_name_offset);
            if (!strncmp(symbol_name, (char*) data_entry->bytes, strlen(symbol_name))) {
                *offset = code_offset;
                return HSA_STATUS_SUCCESS;
            }
        }
        code_offset += code_entry->byteCount;
        code_entry = (BrigBase*) ((char*)code_section_header + code_offset);
    }
    return HSA_STATUS_ERROR;
}
