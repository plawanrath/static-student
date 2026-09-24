/* The linked binary. One payload per input line; prints the struct or "D", then the counters and the certificate to
 * stderr. `--first-struct` stops after one payload, which is what the start-up harness times. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ss_hybrid.h"
#include "ss_payload.h"

int main(int argc, char **argv) {
    double p = 0.0, budget = 1.0;
    uint64_t seed = 1;
    size_t queue = 0;
    int student_first = 0, first_struct = 0, quiet = 0, bench = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--audit") && i + 1 < argc) p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--student-first")) student_first = 1;
        else if (!strcmp(argv[i], "--first-struct")) first_struct = 1;
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--bench")) bench = 1;
        else if (!strcmp(argv[i], "--audit-async") && i + 1 < argc) queue = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--budget") && i + 1 < argc) budget = atof(argv[++i]);
        else if (!strcmp(argv[i], "--certificate")) {
            printf("%s alpha=%g delta=%g tau=%.9g coverage=%.6g n=%u k=%u weights=%s pool=%s grid=%s\n",
                   SS_CERTIFICATE.magic, (double)SS_CERTIFICATE.alpha, (double)SS_CERTIFICATE.delta,
                   (double)SS_CERTIFICATE.tau, (double)SS_CERTIFICATE.coverage, SS_CERTIFICATE.n, SS_CERTIFICATE.k,
                   SS_CERTIFICATE.weights_sha256, SS_CERTIFICATE.pool_sha256, SS_CERTIFICATE.grid_sha256);
            return 0;
        }
    }
    ms_hybrid_configure(p, seed);
    ms_hybrid_set_budget(budget);
    if (queue) ms_hybrid_audit_async(queue);
    metric_struct ms;
    if (first_struct) {                     /* start-to-first-struct: one payload, then exit */
        const char *line = SS_FIRST_PAYLOAD;
        ms_status st = ms_hybrid_parse((const uint8_t *)line, strlen(line), &ms);
        if (!quiet) printf("%s\n", st == MS_OK ? "K" : "D");
        return 0;
    }
    if (bench) {
        /* Read the whole pool into memory first, then time the parse loop alone: no I/O, no allocation, one
         * per-payload timing each, so that p50 and p99 describe the parser and not the harness. */
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
        double *us = malloc(count * sizeof *us);
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (size_t i = 0; i < count; i++) {
            struct timespec s0, s1;
            clock_gettime(CLOCK_MONOTONIC, &s0);
            (student_first ? ms_student_parse : ms_hybrid_parse)((const uint8_t *)pl[i], ln[i], &ms);
            clock_gettime(CLOCK_MONOTONIC, &s1);
            us[i] = (double)(s1.tv_sec - s0.tv_sec) * 1e6 + (double)(s1.tv_nsec - s0.tv_nsec) / 1e3;
        }
        clock_gettime(CLOCK_MONOTONIC, &b);
        double wall = (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) / 1e9;
        for (size_t i = 0; i < count; i++) printf("%.4f\n", us[i]);
        ms_hybrid_audit_stop(); /* joins and drains, so the counters below include every queued sample */
        const ss_counters *cc = ms_hybrid_counters();
        fprintf(stderr, "payloads=%zu wall_s=%.6f throughput_per_s=%.1f legacy_matched=%llu no_match=%llu student_accepted=%llu student_deferred=%llu audited=%llu confident_disagreement=%llu audit_dropped=%llu budget_skipped=%llu\n",
                count, wall, (double)count / wall, (unsigned long long)cc->legacy_matched, (unsigned long long)cc->no_match,
                (unsigned long long)cc->student_accepted, (unsigned long long)cc->student_deferred, (unsigned long long)cc->audited,
                (unsigned long long)cc->confident_disagreement, (unsigned long long)cc->audit_dropped, (unsigned long long)cc->budget_skipped);
        return 0;
    }
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    while ((n = getline(&line, &cap, stdin)) > 0) {
        if (n && line[n - 1] == '\n') n--;
        ms_status st = (student_first ? ms_student_parse : ms_hybrid_parse)((const uint8_t *)line, (size_t)n, &ms);
        if (!quiet) {
            if (st == MS_OK) printf("K\t%llu\t%.*s\n", (unsigned long long)ms.latency_us, (int)ms.user_id_len, ms.user_id);
            else printf("D\n");
        }
    }
    free(line);
    ms_hybrid_audit_stop();
    const ss_counters *c = ms_hybrid_counters();
    fprintf(stderr, "payloads=%llu legacy_matched=%llu no_match=%llu student_accepted=%llu student_deferred=%llu audited=%llu confident_disagreement=%llu\n",
            (unsigned long long)c->payloads, (unsigned long long)c->legacy_matched, (unsigned long long)c->no_match,
            (unsigned long long)c->student_accepted, (unsigned long long)c->student_deferred,
            (unsigned long long)c->audited, (unsigned long long)c->confident_disagreement);
    return 0;
}
