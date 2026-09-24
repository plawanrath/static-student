/* Spans and a unit class -> metric_struct. The network never emits digits: it points at bytes, and this file turns
 * those bytes into a value. It mirrors `to_micros` and `decode` in the Python task definition exactly, including the
 * two bounds that stop a span over a hexadecimal identifier from being read as a number with a huge exponent
 * (at most 40 bytes in the token, at most two digits in the exponent). Arithmetic is exact: the value is built as a
 * ratio of 128-bit integers and floored once, never through floating point. */
#include "ss_struct.h"

#include <string.h>

#define MAX_NUMBER_BYTES 40
/* Two bounds, both chosen so that 128-bit arithmetic cannot wrap. MANT_CAP stops accumulating significant digits
 * early enough that mantissa * 60000000 (the largest unit numerator) stays far below the 128-bit maximum; the digits
 * dropped after it cannot move the floor of a value small enough to fit a uint64. BIG bounds the scaling loops. */
#define MANT_CAP ((unsigned __int128)1e28)
#define BIG ((unsigned __int128)1e30)

static int is_id_byte(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
        || c == '.' || c == '_' || c == ':' || c == '@' || c == '/' || c == '+' || c == '-';
}

/* The unit as an exact ratio of microseconds. */
static void unit_ratio(int unit, unsigned __int128 *num, unsigned __int128 *den) {
    *den = 1;
    switch (unit) {
        case SS_UNIT_NS:  *num = 1; *den = 1000; break;
        case SS_UNIT_US:  *num = 1; break;
        case SS_UNIT_MS:  *num = 1000; break;
        case SS_UNIT_S:   *num = 1000000; break;
        default:          *num = 60000000; break; /* min */
    }
}

/* Accepts  digits[.digits][(e|E)[+|-]dd]  and  1-3 digits followed by groups of exactly three, as thousands
 * separators. Returns 0 and writes the value in microseconds, or -1 if the bytes are not such a number. */
int ss_to_micros(const uint8_t *s, size_t n, int unit, uint64_t *out) {
    if (n == 0 || n > MAX_NUMBER_BYTES) return -1;
    unsigned __int128 mant = 0;
    int frac = 0, exp10 = 0, digits = 0, dropped = 0;
    size_t i = 0;

    if (s[0] == '0' && n > 1 && s[1] == ',') return -1;
    size_t first = 0;
    while (first < n && s[first] >= '0' && s[first] <= '9') first++;
    if (first == 0) return -1;
    if (first < n && s[first] == ',') {           /* thousands groups: 1-3 digits, then ",ddd" repeated */
        if (first > 3 || s[0] == '0') return -1;
        size_t p = first;
        while (p < n && s[p] == ',') {
            if (p + 3 >= n) return -1;
            for (int k = 1; k <= 3; k++) if (s[p + k] < '0' || s[p + k] > '9') return -1;
            p += 4;
        }
        if (p < n && s[p] >= '0' && s[p] <= '9') return -1;
    }
    for (; i < n; i++) {                          /* integer part, commas already validated */
        if (s[i] == ',') continue;
        if (s[i] < '0' || s[i] > '9') break;
        if (mant < MANT_CAP) mant = mant * 10 + (unsigned)(s[i] - '0');
        else dropped++;              /* a dropped integer digit is a factor of ten */
        digits++;
    }
    if (digits == 0) return -1;
    if (i < n && s[i] == '.') {
        i++;
        size_t start = i;
        for (; i < n && s[i] >= '0' && s[i] <= '9'; i++) {
            if (mant < MANT_CAP) { mant = mant * 10 + (unsigned)(s[i] - '0'); frac++; }
        }
        if (i == start) return -1;                /* a trailing dot is not a number */
    }
    if (i < n && (s[i] == 'e' || s[i] == 'E')) {
        i++;
        int sign = 1;
        if (i < n && (s[i] == '+' || s[i] == '-')) { sign = (s[i] == '-') ? -1 : 1; i++; }
        size_t start = i;
        int e = 0;
        for (; i < n && s[i] >= '0' && s[i] <= '9'; i++) e = e * 10 + (s[i] - '0');
        if (i == start || i - start > 2) return -1; /* the exponent is at most two digits */
        exp10 = sign * e;
    }
    if (i != n) return -1;                        /* trailing bytes: not a number */

    unsigned __int128 num, den;
    unit_ratio(unit, &num, &den);
    int e = exp10 + dropped - frac;
    num = num * mant;
    for (; e > 0; e--) { num *= 10; if (num > BIG) return -1; }   /* beyond uint64 microseconds either way */
    for (; e < 0; e++) {
        den *= 10;
        if (den > BIG && den > num) { *out = 0; return 0; }   /* the floor is exactly zero */
    }
    unsigned __int128 v = num / den;
    if (v > (unsigned __int128)UINT64_MAX) return -1;
    *out = (uint64_t)v;
    return 0;
}

int ss_to_struct(const uint8_t *payload, size_t len, const int32_t ptr[4], int unit, metric_struct *out) {
    if (len == 0 || len > SS_MAX_PAYLOAD) return -1;
    int32_t ls = ptr[0], le = ptr[1], us = ptr[2], ue = ptr[3];
    if (ls <= 0 || le <= 0 || us <= 0 || ue <= 0) return -1;         /* position 0 means "field absent" */
    if (le < ls || ue < us) return -1;
    if ((size_t)le > len || (size_t)ue > len) return -1;
    if (unit < 0 || unit >= SS_UNIT_COUNT) return -1;
    size_t id_len = (size_t)(ue - us + 1);
    if (id_len == 0 || id_len > MS_USER_ID_MAX) return -1;
    for (size_t i = 0; i < id_len; i++) if (!is_id_byte(payload[us - 1 + i])) return -1;
    if (ss_to_micros(payload + ls - 1, (size_t)(le - ls + 1), unit, &out->latency_us) != 0) return -1;
    out->user_id_len = (uint8_t)id_len;
    memcpy(out->user_id, payload + us - 1, id_len);
    out->user_id[id_len] = '\0';
    return 0;
}
