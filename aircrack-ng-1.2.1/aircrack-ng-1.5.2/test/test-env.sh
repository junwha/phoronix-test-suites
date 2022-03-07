#!/bin/sh

abs_builddir="/var/lib/phoronix-test-suite/installed-tests/pts/aircrack-ng-1.2.1/aircrack-ng-1.5.2/test"; export abs_builddir
abs_srcdir="/var/lib/phoronix-test-suite/installed-tests/pts/aircrack-ng-1.2.1/aircrack-ng-1.5.2/test"; export abs_srcdir
top_builddir=".."; export top_builddir
top_srcdir=".."; export top_srcdir

EXEEXT=""; export EXEEXT

AIRCRACK_LIBEXEC_PATH="/var/lib/phoronix-test-suite/installed-tests/pts/aircrack-ng-1.2.1/aircrack-ng-1.5.2/src"; export AIRCRACK_LIBEXEC_PATH

AIRCRACK_NG_ARGS="${AIRCRACK_NG_ARGS:--p 4}"; export AIRCRACK_NG_ARGS
