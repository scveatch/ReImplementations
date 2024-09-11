# Variables for compiler and flags
CC = gcc
CFLAGS = -Wall -Iinclude -lm # -I to include header files, -lm for math library

# Target executable name
TARGET = cnn

# Source and object files
SRCDIR = src
OBJDIR = obj
OBJS = $(OBJDIR)/main.o $(OBJDIR)/cnn.o $(OBJDIR)/layers.o $(OBJDIR)/utils.o

# Default rule to build the executable
$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(CFLAGS)

# Rule for each object file
$(OBJDIR)/main.o: $(SRCDIR)/main.c include/cnn.h include/layers.h include/utils.h
	$(CC) $(CFLAGS) -c $(SRCDIR)/main.c -o $(OBJDIR)/main.o

$(OBJDIR)/cnn.o: $(SRCDIR)/cnn.c include/cnn.h include/layers.h
	$(CC) $(CFLAGS) -c $(SRCDIR)/cnn.c -o $(OBJDIR)/cnn.o

$(OBJDIR)/layers.o: $(SRCDIR)/layers.c include/layers.h
	$(CC) $(CFLAGS) -c $(SRCDIR)/layers.c -o $(OBJDIR)/layers.o

$(OBJDIR)/utils.o: $(SRCDIR)/utils.c include/utils.h
	$(CC) $(CFLAGS) -c $(SRCDIR)/utils.c -o $(OBJDIR)/utils.o

# Clean rule to remove compiled files
clean:
	rm -f $(OBJDIR)/*.o $(TARGET)

# Ensure obj directory exists
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Ensure that 'make' creates the obj directory first
$(OBJS): | $(OBJDIR)

