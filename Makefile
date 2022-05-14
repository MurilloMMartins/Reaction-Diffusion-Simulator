CC = g++
INCLUDES = includes
SOURCES = sources/*.cpp
BINARY = ./main
PROG = main.cpp
FLAGS = glad.c -lglfw -lGL -ldl -g -lm
VFLAGS = --show-leak-kinds=all --track-origins=yes --leak-check=full -s

all:
	@$(CC) -o $(BINARY) $(PROG) $(SOURCES) -I$(INCLUDES) $(FLAGS)

run:
	@$(BINARY)

val: all clear 
	valgrind $(VFLAGS) $(BINARY) 

zip:
	zip -r ChaoticParticleSim.zip main.c includes sources Makefile

clean:
	rm $(BINARY)

clear:
	clear
