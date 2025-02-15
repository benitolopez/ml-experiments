CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lm
BUILD_DIR = build
TARGET = $(BUILD_DIR)/iris_classifier

# Object files inside the build directory
OBJ = $(BUILD_DIR)/main.o $(BUILD_DIR)/iris_nn.o

all: $(TARGET)

# Link the executable
$(TARGET): $(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(LDFLAGS)

# Compile main.c into build/main.o
$(BUILD_DIR)/main.o: main.c iris_nn.h | $(BUILD_DIR)
	$(CC) -c main.c -o $@ $(CFLAGS)

# Compile iris_nn.c into build/iris_nn.o
$(BUILD_DIR)/iris_nn.o: iris_nn.c iris_nn.h | $(BUILD_DIR)
	$(CC) -c iris_nn.c -o $@ $(CFLAGS)

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
