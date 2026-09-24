/* Line-oriented driver for the generated student kernel: one payload per input line, one result per output line
 * ("<ptr0> <ptr1> <ptr2> <ptr3> <unit> <score> <defer>"). Used by the parity check and the start-up harness. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ss_kernel.h"

static ss_scratch scratch;

int main(int argc, char **argv) {
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    ss_result r;
    int warm = argc > 1 ? atoi(argv[1]) : 0;
    if (warm) { /* first-struct latency harness: run one payload and exit */
        const char *p = "lat=1ms user=1";
        ss_forward((const uint8_t *)p, strlen(p), &scratch, &r);
        printf("%d\n", r.ptr[0]);
        return 0;
    }
    while ((n = getline(&line, &cap, stdin)) > 0) {
        if (n && line[n - 1] == '\n') n--;
        if (ss_forward((const uint8_t *)line, (size_t)n, &scratch, &r) != 0) return 1;
        printf("%d %d %d %d %d %.9g %.9g\n", r.ptr[0], r.ptr[1], r.ptr[2], r.ptr[3], r.unit, r.score, r.defer_logit);
    }
    free(line);
    return 0;
}
