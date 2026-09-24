/* The linked binary's entry point: the unchanged legacy parser runs first, the student rescues what it rejects, and
 * the two audit each other. Same struct type either way, so a caller cannot tell which path produced a result. */
#ifndef SS_HYBRID_H
#define SS_HYBRID_H

#include <stddef.h>
#include <stdint.h>

#include "metric_struct.h"

typedef struct {
    uint64_t payloads;               /* seen */
    uint64_t legacy_matched;         /* the legacy parser returned a struct */
    uint64_t no_match;               /* it did not: the far-drift signal */
    uint64_t student_accepted;       /* the student rescued one of those, at or above the certified threshold */
    uint64_t student_deferred;       /* it declined: the binary behaves exactly as it would without the student */
    uint64_t audited;                /* matched payloads on which the student was run as an oracle */
    uint64_t confident_disagreement; /* the student was confident and disagreed: the near-drift signal */
    uint64_t audit_dropped;          /* sampled for audit but the queue was full: the audit's own loss rate */
    uint64_t budget_skipped;         /* a no-match the student was not allowed to attempt, under the budget */
} ss_counters;

/* The certificate, as it sits in the binary's read-only data. */
typedef struct {
    char magic[9];        /* "SSCERT01" plus its terminator, so the record can be found in a stripped binary */
    float alpha, delta;   /* the contract: at most alpha wrong among accepted, with confidence 1 - delta */
    float tau;            /* the certified acceptance threshold */
    float coverage;       /* the fraction of the calibration pool it accepted */
    uint32_t n, k;        /* accepted points and errors among them */
    char weights_sha256[65], pool_sha256[65], grid_sha256[65];
} ss_certificate;

extern const ss_certificate SS_CERTIFICATE;

void ms_hybrid_configure(double audit_probability, uint64_t seed);

/* ADR-0010. The student costs about four orders of magnitude more than the legacy parser per payload, so neither the
 * audit nor the rescue path may be allowed to run unbounded on a request path.
 *
 * ms_hybrid_audit_async starts one background thread that drains a fixed-size queue of sampled payloads. The request
 * path then pays only the sampling decision and a bounded copy; when the queue is full the sample is dropped and
 * counted, so the audit degrades by losing samples rather than by slowing the service. Call it after
 * ms_hybrid_configure and call ms_hybrid_audit_stop before reading the counters, which joins the thread and drains.
 *
 * ms_hybrid_set_budget caps the share of payloads on which the student may run at all. Above the cap a no-match is
 * returned as MS_DEFER without consulting the student, which is exactly what the binary would do without one. A
 * budget of 1.0 means unrestricted; 0.0 disables the student entirely. */
void ms_hybrid_audit_async(size_t queue_capacity);
void ms_hybrid_audit_stop(void);
void ms_hybrid_set_budget(double max_student_fraction);
ms_status ms_hybrid_parse(const uint8_t *payload, size_t len, metric_struct *out);
ms_status ms_student_parse(const uint8_t *payload, size_t len, metric_struct *out); /* the student-first ablation */
const ss_counters *ms_hybrid_counters(void);
void ms_hybrid_reset(void);

#endif
