
CC = gcc
CXX = @CXX@
AS = gcc
LD = gcc
CPP = gcc
CFLAGS = -m64 -O3 -march=native  -std=gnu89 -I/usr/local/include -DARCH_LITTLE_ENDIAN=1 -Wall -Wno-stringop-truncation -Wno-format-overflow -Wno-format-truncation -fno-omit-frame-pointer --param allow-store-data-races=0 -Wno-deprecated-declarations -Wformat-extra-args -Wunused-but-set-variable -std=gnu89 -Wdate-time -D_POSIX_SOURCE -D_GNU_SOURCE -D_XOPEN_SOURCE=600 
ASFLAGS = -m64  -DARCH_LITTLE_ENDIAN=1 -c 
LDFLAGS = -m64  -L/usr/local/lib -L/usr/lib64 -L/lib64 -lssl -lcrypto
YASM = /usr/bin/yasm
AR = ar
FIND = /usr/bin/find
RM = /bin/rm -f
GCCV44 := $(shell expr `$(CC) -dumpversion` \>= 4.4)
USE_AESNI = linux

AESIN = aes.o openssl/ossl_aes.o
SUBDIRS = openssl
ifeq "$(GCCV44)" "1"
	ifneq "$(YASM)" ""
		ifdef USE_AESNI
			ifdef AESNI_ARCH
				AESIN += aesni/*.o
				SUBDIRS += aesni
				AESNI_DEC = -DAESNI_IN_USE
			endif
		endif
	endif
endif

aes.o: aes.c ../aes.h aes_func.h
	$(CC) $(CFLAGS) $(AESNI_DEC) -c aes.c -o aes.o

.PHONY: subdirs $(SUBDIRS)

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ all

aes.a: $(SUBDIRS) aes.o
	$(AR) -rs $@ $(AESIN)

default: aes.a
all: aes.a

clean:
	$(FIND) . -name \*.a -exec $(RM) {} \;
	$(FIND) . -name \*.o -exec $(RM) {} \;

distclean: clean
	$(RM) Makefile
	$(RM) aesni/Makefile
	$(RM) openssl/Makefile
