#include "ss_hybrid.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ss_kernel.h"
#include "ss_struct.h"

extern ms_status legacy_parse(const uint8_t *payload, size_t len, metric_struct *out);

static ss_counters C;
static ss_scratch scratch;         /* the request path's */
static ss_scratch audit_scratch;   /* the audit thread's: the two must never share one (ADR-0010) */
static double budget_fraction = 1.0;

/* A single-producer, single-consumer queue of sampled payloads. The request path writes; the audit thread reads. */
typedef struct {
    uint8_t payload[SS_MAX_PAYLOAD];
    size_t len;
    metric_struct legacy;
} audit_item;

static audit_item *queue;
static size_t queue_cap;
static _Atomic size_t q_head, q_tail;
static _Atomic uint64_t a_audited, a_disagree, a_dropped;
static _Atomic int auditor_running;
static pthread_t auditor;
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t audit_cutoff; /* audit iff the next draw is below this */

void ms_hybrid_configure(double audit_probability, uint64_t seed) {
    if (audit_probability < 0.0) audit_probability = 0.0;
    if (audit_probability > 1.0) audit_probability = 1.0;
    audit_cutoff = (uint64_t)(audit_probability * 18446744073709551615.0);
    rng_state = seed ? seed : 0x9E3779B97F4A7C15ull;
}

static uint64_t next_random(void) { /* xorshift64*: the audit sample is reproducible from the seed */
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1Dull;
}

static int student_with(ss_scratch *sc, const uint8_t *payload, size_t len, metric_struct *out, float *score) {
    ss_result r;
    if (ss_forward(payload, len, sc, &r) != 0) return 1;
    *score = r.score;
    if (r.score < SS_CERTIFICATE.tau) return 1;
    return ss_to_struct(payload, len, r.ptr, r.unit, out) == 0 ? 0 : 1;
}

/* Runs the kernel and converts its spans. Returns 0 and fills `out` when the student would accept under the
 * certificate, 1 when it declines, and writes the score for the caller that needs it. */
static int student(const uint8_t *payload, size_t len, metric_struct *out, float *score) {
    return student_with(&scratch, payload, len, out, score);
}

static int same_struct(const metric_struct *a, const metric_struct *b) {
    return a->latency_us == b->latency_us && a->user_id_len == b->user_id_len
        && memcmp(a->user_id, b->user_id, a->user_id_len) == 0;
}

static void *audit_loop(void *unused) {
    (void)unused;
    while (atomic_load_explicit(&auditor_running, memory_order_acquire) ||
           atomic_load_explicit(&q_tail, memory_order_acquire) != atomic_load_explicit(&q_head, memory_order_acquire)) {
        size_t tail = atomic_load_explicit(&q_tail, memory_order_relaxed);
        if (tail == atomic_load_explicit(&q_head, memory_order_acquire)) {
            struct timespec ts = {0, 200000}; /* 0.2 ms: idle politely rather than spin on a shared core */
            nanosleep(&ts, NULL);
            continue;
        }
        const audit_item *it = &queue[tail % queue_cap];
        metric_struct s;
        float score;
        int accepted = student_with(&audit_scratch, it->payload, it->len, &s, &score) == 0;
        atomic_fetch_add_explicit(&a_audited, 1, memory_order_relaxed);
        if (accepted && !same_struct(&s, &it->legacy)) atomic_fetch_add_explicit(&a_disagree, 1, memory_order_relaxed);
        atomic_store_explicit(&q_tail, tail + 1, memory_order_release);
    }
    return NULL;
}

void ms_hybrid_audit_async(size_t queue_capacity) {
    if (atomic_load(&auditor_running) || queue_capacity == 0) return;
    queue_cap = queue_capacity;
    queue = (audit_item *)calloc(queue_cap, sizeof *queue);
    if (!queue) return;
    atomic_store(&q_head, 0);
    atomic_store(&q_tail, 0);
    atomic_store(&auditor_running, 1);
    if (pthread_create(&auditor, NULL, audit_loop, NULL) != 0) { atomic_store(&auditor_running, 0); free(queue); queue = NULL; }
}

void ms_hybrid_audit_stop(void) {
    if (!atomic_load(&auditor_running)) return;
    atomic_store_explicit(&auditor_running, 0, memory_order_release);
    pthread_join(auditor, NULL);
    C.audited += atomic_load(&a_audited);
    C.confident_disagreement += atomic_load(&a_disagree);
    C.audit_dropped += atomic_load(&a_dropped);
    free(queue);
    queue = NULL;
}

void ms_hybrid_set_budget(double max_student_fraction) {
    budget_fraction = max_student_fraction < 0.0 ? 0.0 : (max_student_fraction > 1.0 ? 1.0 : max_student_fraction);
}

/* Queue a copy of a payload the legacy parser accepted, for the audit thread to check later. Bounded work: one
 * comparison, one bounded copy, and a drop if the consumer is behind. */
static void audit_enqueue(const uint8_t *payload, size_t len, const metric_struct *legacy) {
    size_t head = atomic_load_explicit(&q_head, memory_order_relaxed);
    if (head - atomic_load_explicit(&q_tail, memory_order_acquire) >= queue_cap) {
        atomic_fetch_add_explicit(&a_dropped, 1, memory_order_relaxed);
        return;
    }
    audit_item *it = &queue[head % queue_cap];
    if (len > SS_MAX_PAYLOAD) len = SS_MAX_PAYLOAD;
    memcpy(it->payload, payload, len);
    it->len = len;
    it->legacy = *legacy;
    atomic_store_explicit(&q_head, head + 1, memory_order_release);
}

ms_status ms_hybrid_parse(const uint8_t *payload, size_t len, metric_struct *out) {
    C.payloads++;
    if (legacy_parse(payload, len, out) == MS_OK) {
        C.legacy_matched++;
        if (audit_cutoff && next_random() < audit_cutoff) { /* the legacy parser is a free oracle on this population */
            if (queue) {
                audit_enqueue(payload, len, out);
            } else {
                metric_struct s;
                float score;
                C.audited++;
                if (student(payload, len, &s, &score) == 0 && !same_struct(&s, out)) C.confident_disagreement++;
            }
        }
        return MS_OK;
    }
    C.no_match++;
    /* ADR-0010: above the budget the student is not attempted, so a service under drift degrades to today's
     * behaviour instead of collapsing. The comparison is on counts, so the cap is a share of all payloads seen. */
    if ((double)(C.student_accepted + C.student_deferred) >= budget_fraction * (double)C.payloads) {
        C.budget_skipped++;
        return MS_DEFER;
    }
    float score;
    if (student(payload, len, out, &score) == 0) { C.student_accepted++; return MS_OK; }
    C.student_deferred++;
    return MS_DEFER; /* exactly what the binary would have done without the student */
}

ms_status ms_student_parse(const uint8_t *payload, size_t len, metric_struct *out) {
    C.payloads++;
    float score;
    if (student(payload, len, out, &score) == 0) { C.student_accepted++; return MS_OK; }
    C.student_deferred++;
    return legacy_parse(payload, len, out);
}

const ss_counters *ms_hybrid_counters(void) { return &C; }
void ms_hybrid_reset(void) { memset(&C, 0, sizeof C); }
