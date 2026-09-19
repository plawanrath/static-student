/* The struct both paths emit. The student kernel and the legacy parser fill the same type, so the downstream
 * writer cannot tell which path produced it. */
#ifndef METRIC_STRUCT_H
#define METRIC_STRUCT_H

#include <stddef.h>
#include <stdint.h>

#define MS_USER_ID_MAX 36 /* fits decimal, hex and canonical UUID ids */

typedef struct {
    uint64_t latency_us;                  /* latency normalized to microseconds, truncated toward zero */
    uint8_t  user_id_len;                 /* bytes used in user_id */
    char     user_id[MS_USER_ID_MAX + 1]; /* id bytes exactly as they appear in the payload, NUL-terminated */
} metric_struct;

typedef enum { MS_OK = 0, MS_DEFER = 1 } ms_status;

/* Both parsers share this signature. MS_DEFER means "no answer"; *out is then unspecified. */
typedef ms_status (*ms_parse_fn)(const uint8_t *payload, size_t len, metric_struct *out);

#endif
