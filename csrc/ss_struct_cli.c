/* Test driver: each input line is "<unit index>\t<number bytes>"; prints the microsecond value or "X". */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ss_struct.h"

int main(void) {
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    while ((n = getline(&line, &cap, stdin)) > 0) {
        if (n && line[n - 1] == '\n') line[--n] = '\0';
        char *tab = strchr(line, '\t');
        if (!tab) { printf("X\n"); continue; }
        int unit = atoi(line);
        uint64_t v;
        if (ss_to_micros((const uint8_t *)(tab + 1), strlen(tab + 1), unit, &v) == 0) printf("%llu\n", (unsigned long long)v);
        else printf("X\n");
    }
    free(line);
    return 0;
}
