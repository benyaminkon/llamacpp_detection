# Makefile for mmap‐based tiktoken decode_bytes attacker

CC      := gcc
CFLAGS  := -g -O2 -Wall -Isrc
LDFLAGS := 
LIBS    := -lmastik
SRC     := src/attacker.c
BIN_DIR := bin
TARGET  := $(BIN_DIR)/attacker

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)
