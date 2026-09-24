/* Turning the student's spans into the struct both paths emit. */
#ifndef SS_STRUCT_H
#define SS_STRUCT_H

#include <stddef.h>
#include <stdint.h>

#include "metric_struct.h"

#define SS_MAX_PAYLOAD 256
enum { SS_UNIT_NS = 0, SS_UNIT_US, SS_UNIT_MS, SS_UNIT_S, SS_UNIT_MIN, SS_UNIT_COUNT };

/* 0 on success. The span bounds are the kernel's pointer outputs: 1-based, inclusive, 0 meaning "absent". */
int ss_to_micros(const uint8_t *number, size_t n, int unit, uint64_t *out);
int ss_to_struct(const uint8_t *payload, size_t len, const int32_t ptr[4], int unit, metric_struct *out);

#endif
