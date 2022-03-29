
CC = gcc
CXX = @CXX@
AS = gcc
LD = gcc
CPP = gcc
CFLAGS = -m64 -O3 -march=native  -std=gnu89 -I/usr/local/include -DARCH_LITTLE_ENDIAN=1 -Wall -Wno-stringop-truncation -Wno-format-overflow -Wno-format-truncation -fno-omit-frame-pointer --param allow-store-data-races=0 -Wno-deprecated-declarations -Wformat-extra-args -Wunused-but-set-variable -std=gnu89 -Wdate-time -D_POSIX_SOURCE -D_GNU_SOURCE -D_XOPEN_SOURCE=600 
ASFLAGS = -m64  -DARCH_LITTLE_ENDIAN=1 -c 
LDFLAGS = -m64  -L/usr/local/lib -L/usr/lib64 -L/lib64 -lssl -lcrypto

ossl_aes.o: ossl_aes.c ossl_aes.h
	$(CC) $(CFLAGS) -c ossl_aes.c -o $@

default: ossl_aes.o
all: ossl_aes.o

clean:

distclean: clean
	$(RM) Makefile
