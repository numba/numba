#include <stdint.h>
#include <stdlib.h>

typedef struct {
  char *id;
  uintptr_t flags;
  void *ptr;
} PyCustomSlots_Entry;

typedef struct {
  uint64_t flags;
  uint64_t m_f, m_g;
  PyCustomSlots_Entry *entries;
  uint16_t n, b;
  uint8_t r;
  uint8_t reserved;

  /* Trailing variable-size:
  uint16_t d[0];
  PyCustomSlots_Entry entries_mem[n];
  */
} PyCustomSlots_Table;

typedef struct {
  PyCustomSlots_Table base;
  uint16_t d[64];
  PyCustomSlots_Entry entries_mem[64];
} PyCustomSlots_Table_64_64;



/*void PyCustomSlots_bucket_argsort(uint16_t p, uint16_t) {
  }*/


uint64_t PyCustomSlots_roundup_2pow(uint64_t x) {
  x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8;
  x |= x >> 16; x |= x >> 32;
  return x + 1;
}

#define BIN_LIMIT 8

  
void _PyCustomSlots_bucket_argsort(uint16_t *p, uint8_t *binsizes,
                                   uint8_t *number_of_bins_by_size) {
  uint16_t *sort_bins[BIN_LIMIT];
  int binsize, ibin, nbins;
  nbins = 0;
  for (binsize = BIN_LIMIT - 1; binsize >= 0; --binsize) {
    sort_bins[binsize] = p;
    p += number_of_bins_by_size[binsize];
    nbins += number_of_bins_by_size[binsize];
  }
  for (ibin = 0; ibin != nbins; ++ibin) {
    binsize = binsizes[ibin];
    *sort_bins[binsize] = ibin;
    sort_bins[binsize]++;
  }
}



int PyCustomSlots_PerfectHash(PyCustomSlots_Table *table, uint64_t *hashes) {
  uint16_t bin, j;
  uint8_t binsize;
  uint16_t i, n = table->n, b = table->b;
  uint64_t m_f = PyCustomSlots_roundup_2pow(table->n) - 1;
  uint64_t m_g = (b - 1) & 0xffff;
  uint64_t *bin_hashes = malloc(sizeof(uint64_t) * b * BIN_LIMIT);
  uint64_t *bin_data = malloc(sizeof(uint16_t) * b * BIN_LIMIT);
  uint8_t *binsizes = malloc(sizeof(uint8_t) * b);
  uint16_t *p = malloc(sizeof(uint16_t) * b);
  uint8_t *taken = malloc(sizeof(uint8_t) * n);
  uint8_t number_of_bins_by_size[BIN_LIMIT];
  uint16_t *d = (void*)((char*)table + sizeof(PyCustomSlots_Table));
  PyCustomSlots_Entry *entries_copy = malloc(sizeof(PyCustomSlots_Entry) * n);

  for (i = 0; i != n; ++i) {
    entries_copy[i] = table->entries[i];
  }
  
  /* Bin the n hashes into b bins based on the g hash. Also count the
     number of bins of each size. */
  for (bin = 0; bin != b; ++bin) {
    binsizes[bin] = 0;
  }
  number_of_bins_by_size[0] = b;
  for (binsize = 1; binsize != BIN_LIMIT; ++binsize) {
    number_of_bins_by_size[binsize] = 0;
  }
  for (i = 0; i != n; ++i) {
    bin = hashes[i] & m_g;
    binsize = ++binsizes[bin];
    if (binsize == BIN_LIMIT) {
      printf("ERROR 1\n");
      return -1;
    }
    bin_hashes[BIN_LIMIT * bin + binsize - 1] = hashes[i];
    bin_data[BIN_LIMIT * bin + binsize - 1] = i;
    number_of_bins_by_size[binsize - 1]--;
    number_of_bins_by_size[binsize]++;
  }

  /* argsort the bins (p stores permutation) from largest to
     smallest, using binsort */

  /* Find perfect table -- try again for each choice of r */
  int r;
  for (r = 64; r != -1; --r) {
    printf("r==================================================%d\n", r);
    /* Step 1: Validate that f is 1:1 in each bin */
    for (j = 0; j != b; ++j) {
      int k, t;
      bin = p[j];
      for (k = 0; k != binsizes[bin]; ++k) {
        for (t = k + 1; t <= binsizes[bin]; ++t) {
          if (((bin_hashes[BIN_LIMIT * bin + k] >> r) & m_f) ==
              ((bin_hashes[BIN_LIMIT * bin + t] >> r) & m_f)) {
            goto next_r;
          }
        }
      }
    }

    /* Step 2: Attempt to assign displacements d[bin], starting with
       the largest bin */
    for (i = 0; i != n; ++i) {
      taken[i] = 0;
    }
    for (j = 0; j != b; ++j) {
      uint16_t dval;
      bin = p[j];
      if (binsizes[bin] == 0) {
        d[bin] = 0;
        break;
      }
      for (dval = 0; dval != b; ++dval) {
        printf("%d dval %d\n", bin, dval);
        int k;
        int slots_not_taken = 1;
        for (k = 0; k != binsizes[bin]; ++k) {
          uint16_t slot = (((bin_hashes[BIN_LIMIT * bin + k] >> r) & m_f) ^
                           dval);
          if (taken[slot]) {
            slots_not_taken = 0;
            break;
          }
        }
        if (slots_not_taken) {
          printf("BREAKOUT\n");
          break;
        }
      }
      if (dval == b) {
        /* no appropriate dval found */
        goto next_r;
      } else {
        int k;
        /* mark slots as taken and shuffle in table elements */
        for (k = 0; k != binsizes[bin]; ++k) {
          uint16_t slot = (((bin_hashes[BIN_LIMIT * bin + k] >> r) & m_f) ^
                           dval);
          taken[slot] = 1;
          printf(">>%d", bin_data[BIN_LIMIT * bin + k]);
          table->entries[slot] = entries_copy[bin_data[BIN_LIMIT * bin + k]];
        }
        /* record dval */
        printf(">>>dval%d\n", dval);
        d[bin] = dval;
      }
    }

    break;
  next_r:
    continue;
  }
  table->r = r;
  printf("%d\n", table->r);
  return 0;
}
