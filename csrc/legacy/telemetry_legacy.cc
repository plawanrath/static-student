// Legacy telemetry parser: an ordered list of RE2 rules, one per known source format, first full extraction wins.
// Written against the T=0 source formats in data/drift/sources_v0.json and frozen (see csrc/legacy/FREEZE).
// It is the fallback the hybrid binary links against, unchanged; nothing in it knows about the student.
#include <re2/re2.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include "../metric_struct.h"
}

namespace {

constexpr size_t kMaxPayload = 256;

// A rule extracts both fields or nothing. `line` captures everything in one anchored pattern (positional and
// free-text sources); otherwise `lat` and `usr` are searched independently (keyed sources, any field order).
// Named groups: n = number, u = unit (optional; else `unit` applies), id = user id.
struct RuleSpec {
    const char *name, *line, *lat, *usr, *unit;
};

#define NUM "(?P<n>\\d+(?:\\.\\d+)?)"
#define TS_SYSLOG "[A-Z][a-z]{2} [ \\d]\\d \\d\\d:\\d\\d:\\d\\d"
#define TS_ISO "\\d{4}-\\d\\d-\\d\\dT[\\d:.]+Z"

const RuleSpec kRules[] = {
    {"nginx_kv",
     "^" TS_SYSLOG " \\S+ nginx: method=[A-Z]+ path=\\S+ status=\\d{3} request_time=" NUM " user=(?P<id>\\d+) bytes=\\d+$",
     nullptr, nullptr, "s"},
    {"app_logfmt", nullptr, "(?:^| )duration=" NUM "(?P<u>us|ms|s)(?: |$)", "(?:^| )user_id=(?P<id>[\\w.:\\-]+)(?: |$)", nullptr},
    {"edge_json", nullptr, "\"latency_ms\"\\s*:\\s*" NUM "\\s*[,}]", "\"user\"\\s*:\\s*\"(?P<id>[^\"]+)\"", "ms"},
    {"gateway_json_nested", nullptr, "\"elapsed_us\"\\s*:\\s*" NUM "\\s*[,}]",
     "\"user\"\\s*:\\s*\\{\\s*\"id\"\\s*:\\s*\"(?P<id>[^\"]+)\"", "us"},
    {"java_kv", nullptr, "(?:\\] |; )rt: " NUM " (?P<u>us|ms|s)(?:;|$)", "(?:\\] |; )uid: (?P<id>[\\w.:\\-]+)(?:;|$)", nullptr},
    {"access_qs", nullptr, "[?&]took=" NUM "(?:&| |$)", "[?&]uid=(?P<id>[\\w.:\\-]+)(?:&| |$)", "ms"},
    {"batch_csv", "^" TS_ISO ",[A-Z]+,[^,]*,\\d{3}," NUM "(?P<u>us|ms|s),(?P<id>[\\w.:\\-]+)$", nullptr, nullptr, nullptr},
    {"worker_text", "^" TS_ISO " \\w+ [\\w.\\-]+: handled \\S+ for user (?P<id>[\\w.:\\-]+) in " NUM " (?P<u>us|ms|s)$",
     nullptr, nullptr, nullptr},
    {"metrics_pipe", nullptr, "\\|elapsed_ms=" NUM "(?:\\||$)", "\\|account=(?P<id>[\\w.:\\-]+)(?:\\||$)", "ms"},
    {"quoted_kv", nullptr, " time=\"" NUM "(?P<u>us|ms|s)\"", " user=\"(?P<id>[^\"]+)\"", nullptr},
    {"proxy_text",
     "^" TS_SYSLOG " \\S+ proxy\\[\\d+\\]: user (?P<id>[\\w.:\\-]+) [A-Z]+ \\S+ -> \\d{3} \\(" NUM "(?P<u>us|ms|s)\\)$",
     nullptr, nullptr, nullptr},
    {"audit_json", nullptr, "\"duration\"\\s*:\\s*\"" NUM "(?P<u>us|ms|s)\"", "\"principal\"\\s*:\\s*\"(?P<id>[^\"]+)\"", nullptr},
};

struct Compiled {
    std::unique_ptr<re2::RE2> re;
    int n = -1, u = -1, id = -1;
    explicit Compiled(const char *pattern) {
        if (!pattern) return;
        re = std::make_unique<re2::RE2>(pattern, re2::RE2::Latin1);
        for (const auto &g : re->NamedCapturingGroups()) {
            if (g.first == "n") n = g.second;
            if (g.first == "u") u = g.second;
            if (g.first == "id") id = g.second;
        }
    }
};

struct Rule {
    Compiled line, lat, usr;
    const char *unit;
    explicit Rule(const RuleSpec &s) : line(s.line), lat(s.lat), usr(s.usr), unit(s.unit) {}
};

const std::vector<Rule> &rules() {
    static const std::vector<Rule> *r = [] {
        auto *v = new std::vector<Rule>();
        for (const auto &s : kRules) v->emplace_back(s);
        return v;
    }();
    return *r;
}

// digits[.digits] in `unit` -> microseconds, truncated toward zero, exact in 128-bit integers.
bool to_micros(re2::StringPiece num, re2::StringPiece unit, uint64_t *out) {
    unsigned __int128 mant = 0;
    int frac = 0, digits = 0;
    bool seen_dot = false;
    for (char c : num) {
        if (c == '.') { seen_dot = true; continue; }
        if (++digits > 30) return false;
        mant = mant * 10 + static_cast<unsigned>(c - '0');
        if (seen_dot) ++frac;
    }
    unsigned __int128 mul = 1, div = 1;
    if (unit == "us") mul = 1;
    else if (unit == "ms") mul = 1000;
    else if (unit == "s") mul = 1000000;
    else return false;
    for (int i = 0; i < frac; ++i) div *= 10;
    unsigned __int128 v = mant * mul / div;
    if (v > UINT64_MAX) return false;
    *out = static_cast<uint64_t>(v);
    return true;
}

bool search(const Compiled &c, re2::StringPiece text, std::vector<re2::StringPiece> *m) {
    m->assign(c.re->NumberOfCapturingGroups() + 1, re2::StringPiece());
    return c.re->Match(text, 0, text.size(), re2::RE2::UNANCHORED, m->data(), static_cast<int>(m->size()));
}

}  // namespace

extern "C" ms_status legacy_parse(const uint8_t *payload, size_t len, metric_struct *out) {
    if (len == 0 || len > kMaxPayload) return MS_DEFER;
    re2::StringPiece text(reinterpret_cast<const char *>(payload), len);
    std::vector<re2::StringPiece> m;
    for (const Rule &r : rules()) {
        re2::StringPiece num, unit, id;
        if (r.line.re) {
            if (!search(r.line, text, &m)) continue;
            num = m[r.line.n];
            id = m[r.line.id];
            unit = r.line.u >= 0 ? m[r.line.u] : re2::StringPiece(r.unit);
        } else {
            if (!search(r.lat, text, &m)) continue;
            num = m[r.lat.n];
            unit = r.lat.u >= 0 ? m[r.lat.u] : re2::StringPiece(r.unit);
            if (!search(r.usr, text, &m)) continue;
            id = m[r.usr.id];
        }
        if (id.empty() || id.size() > MS_USER_ID_MAX) continue;
        if (!to_micros(num, unit, &out->latency_us)) continue;
        out->user_id_len = static_cast<uint8_t>(id.size());
        std::memcpy(out->user_id, id.data(), id.size());
        out->user_id[id.size()] = '\0';
        return MS_OK;
    }
    return MS_DEFER;
}
