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
  /* We know how many bins there are of each size, so place pointers
     for each size along on the output array p */
  for (binsize = BIN_LIMIT - 1; binsize >= 0; --binsize) {
    sort_bins[binsize] = p;
    p += number_of_bins_by_size[binsize];
    nbins += number_of_bins_by_size[binsize];
  }
  /* Then simply write indices to the bins */
  for (ibin = 0; ibin != nbins; ++ibin) {
    binsize = binsizes[ibin];
    sort_bins[binsize][0] = ibin;
    sort_bins[binsize]++;
  }
}

int _PyCustomSlots_FindDisplacements(PyCustomSlots_Table *table,
                                     uint64_t *hashes,
                                     uint8_t *binsizes,
                                     uint16_t *bins,
                                     uint16_t *p,
                                     uint8_t *taken,
                                     PyCustomSlots_Entry *entries_copy) {
  uint16_t *d = (void*)((char*)table + sizeof(PyCustomSlots_Table));
  uint16_t nbins = table->b;
  uint64_t m_f = table->m_f;
  uint8_t r = table->r;
  int i, j, bin;

  /* Step 1: Validate that f is 1:1 in each bin */
  for (j = 0; j != nbins; ++j) {
    int k, t;
    bin = p[j];
    for (k = 0; k != binsizes[bin]; ++k) {
      for (t = k + 1; t < binsizes[bin]; ++t) {
        if (((hashes[bins[BIN_LIMIT * bin + k]] >> r) & m_f) ==
            ((hashes[bins[BIN_LIMIT * bin + t]] >> r) & m_f)) {
          return -1;
        }
      }
    }
  }

  /* Step 2: Attempt to assign displacements d[bin], starting with
     the largest bin */
  for (i = 0; i != nbins; ++i) {
    taken[i] = 0;
  }
  for (j = 0; j != nbins; ++j) {
    uint16_t dval;
    bin = p[j];
    if (binsizes[bin] == 0) {
      d[bin] = 0;
    } else {
      for (dval = 0; dval != nbins; ++dval) {
        int k;
        int collides = 0;
        for (k = 0; k != binsizes[bin]; ++k) {
          uint16_t slot = (((hashes[bins[BIN_LIMIT * bin + k]] >> r) & m_f) ^
                           dval);
          if (taken[slot]) {
            collides = 1;
            break;
          }
        }
        if (!collides) break;
      }
      if (dval == nbins) {
        /* no appropriate dval found */
        return -1;
      } else {
        int k;
        /* mark slots as taken and shuffle in table elements */
        for (k = 0; k != binsizes[bin]; ++k) {
          uint16_t slot = (((hashes[bins[BIN_LIMIT * bin + k]] >> r) & m_f) ^
                           dval);
          taken[slot] = 1;
          table->entries[slot] = entries_copy[bins[BIN_LIMIT * bin + k]];
        }
        /* record dval */
        d[bin] = dval;
      }
    }
  }
  return 0;
}

int PyCustomSlots_PerfectHash(PyCustomSlots_Table *table, uint64_t *hashes) {
  uint16_t bin, j;
  uint8_t binsize;
  uint16_t i, n = table->n, b = table->b;
  uint64_t m_f = PyCustomSlots_roundup_2pow(table->n) - 1;
  uint64_t m_g = (b - 1) & 0xffff;
  uint16_t *bins = malloc(sizeof(uint16_t) * b * BIN_LIMIT);
  uint8_t *binsizes = malloc(sizeof(uint8_t) * b);
  uint16_t *p = malloc(sizeof(uint16_t) * b);
  uint8_t *taken = malloc(sizeof(uint8_t) * n);
  uint8_t number_of_bins_by_size[BIN_LIMIT];
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
    bins[BIN_LIMIT * bin + binsize - 1] = i;
    number_of_bins_by_size[binsize - 1]--;
    number_of_bins_by_size[binsize]++;
  }

  /* argsort the bins (p stores permutation) from largest to
     smallest, using binsort */
  _PyCustomSlots_bucket_argsort(p, binsizes, &number_of_bins_by_size[0]);

  /* Find perfect table -- try again for each choice of r */
  table->m_f = m_f;
  table->m_g = m_g;
  int r, retcode;
  for (r = 64; r != -1; --r) {
    table->r = r;
    retcode = _PyCustomSlots_FindDisplacements(table, hashes, binsizes, bins, p,
                                               taken, entries_copy);
    if (retcode == 0) {
      break;
    }
  }
  return 0;
}
