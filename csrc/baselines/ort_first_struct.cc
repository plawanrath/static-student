/* Start-to-first-struct for the same student served by ONNX Runtime, in a C++ host process, so that the comparison is
 * process start to first parsed struct on both sides and not a Python import time. The graph ends at the same three
 * heads the generated kernel computes; the struct conversion and the legacy fallback are the same objects the hybrid
 * binary links, so the only difference being measured is how the model is carried and executed. */
#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "metric_struct.h"
#include "ss_payload.h"
#include "ss_struct.h"
ms_status legacy_parse(const uint8_t *payload, size_t len, metric_struct *out);
}

#ifndef SS_ORT_MAX_LEN
#define SS_ORT_MAX_LEN 257
#endif

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.onnx [--quiet]\n", argv[0]); return 2; }
    const bool quiet = argc > 2 && !strcmp(argv[2], "--quiet");

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "ss");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, argv[1], opts);

    const char *payload = SS_FIRST_PAYLOAD;
    const size_t len = strlen(payload);
    std::vector<int64_t> ids(SS_ORT_MAX_LEN, 256);
    ids[0] = 257;
    for (size_t i = 0; i < len && i + 1 < SS_ORT_MAX_LEN; i++) ids[i + 1] = (unsigned char)payload[i];

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape{1, SS_ORT_MAX_LEN};
    Ort::Value in = Ort::Value::CreateTensor<int64_t>(mem, ids.data(), ids.size(), shape.data(), shape.size());
    const char *in_names[] = {"ids"};
    const char *out_names[] = {"pointers", "unit", "defer"};
    auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 3);

    const float *ptr = outs[0].GetTensorData<float>();   /* (1, 4, T) */
    const float *unit = outs[1].GetTensorData<float>();
    int32_t arg[4];
    for (int r = 0; r < 4; r++) {
        int best = 0;
        for (int t = 1; t < SS_ORT_MAX_LEN; t++)
            if (ptr[(size_t)r * SS_ORT_MAX_LEN + t] > ptr[(size_t)r * SS_ORT_MAX_LEN + best]) best = t;
        arg[r] = best;
    }
    int ua = 0;
    for (int i = 1; i < 5; i++) if (unit[i] > unit[ua]) ua = i;

    metric_struct ms;
    ms_status st = (ss_to_struct((const uint8_t *)payload, len, arg, ua, &ms) == 0) ? MS_OK : MS_DEFER;
    if (st != MS_OK) st = legacy_parse((const uint8_t *)payload, len, &ms);
    if (!quiet) printf("%s\n", st == MS_OK ? "K" : "D");
    return 0;
}
