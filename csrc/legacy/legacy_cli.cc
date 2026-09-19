// Line-oriented driver for a parser with the shared signature: one payload per input line, one result per output
// line ("D" for MS_DEFER, "K\t<latency_us>\t<user_id>" for MS_OK). Used by the Python harness and the tests.
#include <cstdio>
#include <iostream>
#include <string>

extern "C" {
#include "../metric_struct.h"
ms_status legacy_parse(const uint8_t *payload, size_t len, metric_struct *out);
}

int main() {
    std::ios::sync_with_stdio(false);
    std::string line, out;
    metric_struct ms;
    while (std::getline(std::cin, line)) {
        if (legacy_parse(reinterpret_cast<const uint8_t *>(line.data()), line.size(), &ms) == MS_OK) {
            out.append("K\t").append(std::to_string(ms.latency_us)).append("\t").append(ms.user_id, ms.user_id_len).append("\n");
        } else {
            out.append("D\n");
        }
        if (out.size() > (1u << 16)) { fwrite(out.data(), 1, out.size(), stdout); out.clear(); }
    }
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
