CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
LDFLAGS = -lm

SRCS = main.c neural_network.c mnist_loader.c
BUILD_DIR = build
OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/mnist_nn

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
