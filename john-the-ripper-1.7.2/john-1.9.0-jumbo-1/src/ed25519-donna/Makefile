
CC = gcc
CXX = @CXX@
AS = gcc
LD = gcc
CPP = gcc
CPPFLAGS =  -I/usr/local/include -DCL_SILENCE_DEPRECATION
CFLAGS = -m64 -O3 -march=native  -std=gnu89 -I/usr/local/include -DARCH_LITTLE_ENDIAN=1 -Wall -Wno-stringop-truncation -Wno-format-overflow -Wno-format-truncation -fno-omit-frame-pointer --param allow-store-data-races=0 -Wno-deprecated-declarations -Wformat-extra-args -Wunused-but-set-variable -std=gnu89 -Wdate-time -D_POSIX_SOURCE -D_GNU_SOURCE -D_XOPEN_SOURCE=600  -Wno-unused-function
ASFLAGS = -m64  -DARCH_LITTLE_ENDIAN=1 -c 
LDFLAGS = -m64  -L/usr/local/lib -L/usr/lib64 -L/lib64 -lssl -lcrypto
YASM = /usr/bin/yasm
AR = ar
FIND = /usr/bin/find
RM = /bin/rm -f

IN = ed25519-donna.o

all: ed25519-donna.a
default: ed25519-donna.a

ed25519-donna.o: ed25519-donna.c
	$(CC) -DHAVE_CONFIG_H $(CFLAGS) $(CPPFLAGS) -c ed25519-donna.c -o ed25519-donna.o

.PHONY: subdirs $(SUBDIRS)

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ all

ed25519-donna.a: $(SUBDIRS) ed25519-donna.o
	$(AR) -rs $@ $(IN)

clean:
	$(FIND) . -name \*.a -exec $(RM) {} \;
	$(FIND) . -name \*.o -exec $(RM) {} \;

distclean: clean
	$(RM) Makefile
