#include <string.h>
#include <stdio.h>
#include <stdlib.h>


enum {BITS=8};

void
count_occurence(char input[], unsigned freq[], unsigned count)
{
	for (unsigned i=0; i<count; ++i) {
		freq[input[i]] += 1;
	}
}

void scan(unsigned freq[], unsigned offset[])
{
	unsigned cum = 0;
	for (unsigned i=0; i<BITS; ++i) {
		offset[i] = cum;
		cum += freq[i];
	}
}

void final_pos(char input[], unsigned offset[], unsigned indices[], unsigned count)
{
	unsigned bucket_offets[BITS] = {0};

	for (unsigned i=0; i<count; ++i) {
		unsigned off = bucket_offets[input[i]]++;
		unsigned base = offset[input[i]];
		indices[i] = base + off;
	}
}

void scatter(char input[], unsigned indices[], char output[], unsigned count)
{
	for (unsigned i=0; i<count; ++i) {
		output[indices[i]] = input[i];
	}
}

int main() {
	char data[] = {0, 2, 1, 4, 6, 5, 2, 4, 7};

	unsigned count = sizeof(data) / sizeof(char);
	char output[count];

	unsigned freq[BITS];
	unsigned offset[BITS];
	unsigned indices[BITS];
	memset(freq, 0, sizeof(freq));

	count_occurence(data, freq, count);
	scan(freq, offset);
	//final_pos(data, offset, indices, count);
	//scatter(data, indices, output, count);

	puts("offset");
	for(unsigned i=0; i<BITS; ++i) {
		printf("[%u] = %u\n", i, offset[i]);
	}

	// puts("output");
	// for(unsigned i=0; i<count; ++i) {
	// 	printf("[%u] = %u\n", i, output[i]);
	// }
}