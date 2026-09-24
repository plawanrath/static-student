/* The same benchmark loop for the legacy parser alone: the throughput the hybrid has to stay close to. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "metric_struct.h"

extern ms_status legacy_parse(const uint8_t *payload, size_t len, metric_struct *out);

int main(void) {
    size_t cap_n = 1 << 12, count = 0;
    char **pl = malloc(cap_n * sizeof *pl);
    size_t *ln = malloc(cap_n * sizeof *ln);
    char *buf = NULL;
    size_t bcap = 0;
    ssize_t got;
    while ((got = getline(&buf, &bcap, stdin)) > 0) {
        if (got && buf[got - 1] == '\n') got--;
        if (count == cap_n) { cap_n *= 2; pl = realloc(pl, cap_n * sizeof *pl); ln = realloc(ln, cap_n * sizeof *ln); }
        pl[count] = malloc((size_t)got + 1);
        memcpy(pl[count], buf, (size_t)got);
        pl[count][got] = '\0';
        ln[count++] = (size_t)got;
    }
    free(buf);
    metric_struct ms;
    unsigned long long ok = 0;
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (size_t i = 0; i < count; i++) {
        struct timespec s0, s1;
        clock_gettime(CLOCK_MONOTONIC, &s0);
        if (legacy_parse((const uint8_t *)pl[i], ln[i], &ms) == MS_OK) ok++;
        clock_gettime(CLOCK_MONOTONIC, &s1);
        printf("%.4f\n", (double)(s1.tv_sec - s0.tv_sec) * 1e6 + (double)(s1.tv_nsec - s0.tv_nsec) / 1e3);
    }
    clock_gettime(CLOCK_MONOTONIC, &b);
    double wall = (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) / 1e9;
    fprintf(stderr, "payloads=%zu wall_s=%.6f throughput_per_s=%.1f legacy_matched=%llu\n", count, wall, (double)count / wall, ok);
    return 0;
}
