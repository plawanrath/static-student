/* The floor every start-up number is measured against: a binary that links the same legacy parser and produces one
 * struct, with no student, no weights and no kernel. The difference between this and the hybrid is what carrying a
 * model inside the executable costs. */
#include <stdio.h>
#include <string.h>

#include "metric_struct.h"
#include "ss_payload.h"

extern ms_status legacy_parse(const uint8_t *payload, size_t len, metric_struct *out);

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    const char *line = SS_FIRST_PAYLOAD;
    metric_struct ms;
    ms_status st = legacy_parse((const uint8_t *)line, strlen(line), &ms);
    printf("%s\n", st == MS_OK ? "K" : "D");
    return 0;
}
