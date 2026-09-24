# Native pieces. `make -C csrc` builds everything into build/ at the repository root.
BUILD   ?= ../build
CXX     ?= c++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra $(shell pkg-config --cflags re2)
LDLIBS   = $(shell pkg-config --libs re2)

all: $(BUILD)/legacy_telemetry_cli $(BUILD)/ss_struct_cli

$(BUILD)/legacy_telemetry.o: legacy/telemetry_legacy.cc metric_struct.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/legacy_telemetry_cli: legacy/legacy_cli.cc $(BUILD)/legacy_telemetry.o
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/ss_struct_cli: ss_struct_cli.c ss_struct.c ss_struct.h metric_struct.h
	@mkdir -p $(BUILD)
	cc -O3 -std=c11 -Wall -Wextra -I. ss_struct_cli.c ss_struct.c -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
