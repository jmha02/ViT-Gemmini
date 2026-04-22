# Makefile for ivit custom ops — RISC-V Linux cross-compilation
#
# Usage (from ONNXRuntime/ort_ivit_ops/):
#   make ORT_RISCV_DIR=<path>
#
# ORT_RISCV_DIR defaults to the standard onnxruntime-riscv location
# inside the flexi repository.

REPO_ROOT ?= $(abspath ../..)
ORT_RISCV_DIR ?= $(REPO_ROOT)/tvm-gemmini/3rdparty/gemmini/software/onnxruntime-riscv
BUILD_DIR ?= $(REPO_ROOT)/build/ort/ort_ivit_ops

RISCV_CC ?= riscv64-unknown-linux-gnu-gcc
RISCV_CXX ?= riscv64-unknown-linux-gnu-g++
RISCV_AR ?= riscv64-unknown-linux-gnu-ar
HOST_CC ?= cc
HOST_CXX ?= c++

COMMON_INCLUDES = \
	-I$(ORT_RISCV_DIR)/include/onnxruntime/core/session \
	-I$(ORT_RISCV_DIR)/include/onnxruntime

SYSTOLIC_INCLUDES = \
	-I$(ORT_RISCV_DIR)/onnxruntime/core/mlas/lib/systolic

RISCV_CFLAGS = \
	-O2 \
	-march=rv64imafdc \
	-mabi=lp64d \
	-static \
	-std=c11 \
	$(COMMON_INCLUDES)

RISCV_CXXFLAGS = \
	-O2 \
	-march=rv64imafdc \
	-mabi=lp64d \
	-static \
	-std=c++17 \
	$(COMMON_INCLUDES) \
	$(SYSTOLIC_INCLUDES)

HOST_CFLAGS = \
	-O2 \
	-fPIC \
	-std=c11 \
	$(COMMON_INCLUDES)

HOST_CXXFLAGS = \
	-O2 \
	-fPIC \
	-std=c++17 \
	$(COMMON_INCLUDES)

RISCV_OBJS = $(BUILD_DIR)/ivit_ops.riscv.o $(BUILD_DIR)/ivit_gemmini_ops.riscv.o
HOST_OBJS = $(BUILD_DIR)/ivit_ops.host.o $(BUILD_DIR)/ivit_gemmini_ops.host.o
LIB  = $(BUILD_DIR)/libivit_ops.a
HOST_LIB = $(BUILD_DIR)/libivit_ops_host.so

all: $(LIB)

host: $(HOST_LIB)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/ivit_ops.riscv.o: ivit_ops.c ivit_gemmini_ops.h | $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/ivit_gemmini_ops.riscv.o: ivit_gemmini_ops.cc ivit_gemmini_ops.h | $(BUILD_DIR)
	$(RISCV_CXX) $(RISCV_CXXFLAGS) -DIVIT_USE_GEMMINI -c $< -o $@

$(BUILD_DIR)/ivit_ops.host.o: ivit_ops.c ivit_gemmini_ops.h | $(BUILD_DIR)
	$(HOST_CC) $(HOST_CFLAGS) -fPIC -c $< -o $@

$(BUILD_DIR)/ivit_gemmini_ops.host.o: ivit_gemmini_ops.cc ivit_gemmini_ops.h | $(BUILD_DIR)
	$(HOST_CXX) $(HOST_CXXFLAGS) -c $< -o $@

$(LIB): $(RISCV_OBJS)
	$(RISCV_AR) rcs $@ $^

$(HOST_LIB): $(HOST_OBJS)
	mkdir -p $(dir $@)
	$(HOST_CXX) -shared -o $@ $^

clean:
	rm -f $(RISCV_OBJS) $(HOST_OBJS) $(LIB) $(HOST_LIB)

.PHONY: all clean host
