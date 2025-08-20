# Top-level Makefile for tiktoken_effect_l1
# Integrates CacheSC and Mastik with separate binaries

# Paths
CACHESC_INCLUDE := /home/user/Projects/CacheSC/include
CACHESC_LIBDIR  := /home/user/Projects/CacheSC/lib
SRC_DIR         := ./
BIN_DIR         := ../bin

# Compiler and flags
CC      := gcc
CFLAGS  := -g -O1 -fno-tree-vectorize -std=gnu99 \
           -I include -I $(CACHESC_INCLUDE)
LDFLAGS := -L$(CACHESC_LIBDIR)
LDLIBS  := -lcachesc -ldwarf -lelf -lbfd -lpthread -lmastik

# Targets
TARGET_ATTACKER      := $(BIN_DIR)/orchastration_attacker
TARGET_ORCHASTRACTOR := $(BIN_DIR)/orchastractor

# Object files (separate)
ATTACKER_OBJ      := $(BIN_DIR)/orchastration_attacker.o
ORCHASTRACTOR_OBJ := $(BIN_DIR)/orchastractor.o

# Default target
all: $(TARGET_ATTACKER) $(TARGET_ORCHASTRACTOR)

# Ensure binary output directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile source files
$(ATTACKER_OBJ): $(SRC_DIR)/attacker.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(ORCHASTRACTOR_OBJ): $(SRC_DIR)/orchastractor.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link binaries
$(TARGET_ATTACKER): $(ATTACKER_OBJ) | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@ $(LDLIBS)

$(TARGET_ORCHASTRACTOR): $(ORCHASTRACTOR_OBJ) | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@ $(LDLIBS)

# Clean up build artifacts
clean:
	rm -rf $(BIN_DIR)/*.o $(TARGET_ATTACKER) $(TARGET_ORCHASTRACTOR)

.PHONY: all clean
