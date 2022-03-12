# CMake generated Testfile for 
# Source directory: /var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git
# Build directory: /var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(VerifyAll "SMHasher" "--test=VerifyAll")
set_tests_properties(VerifyAll PROPERTIES  _BACKTRACE_TRIPLES "/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;488;add_test;/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;0;")
add_test(Sanity "SMHasher" "--test=Sanity")
set_tests_properties(Sanity PROPERTIES  _BACKTRACE_TRIPLES "/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;489;add_test;/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;0;")
add_test(Speed "SMHasher" "--test=Speed")
set_tests_properties(Speed PROPERTIES  _BACKTRACE_TRIPLES "/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;490;add_test;/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;0;")
add_test(Cyclic "SMHasher" "--test=Cyclic")
set_tests_properties(Cyclic PROPERTIES  _BACKTRACE_TRIPLES "/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;491;add_test;/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;0;")
add_test(Zeroes "SMHasher" "--test=Zeroes")
set_tests_properties(Zeroes PROPERTIES  _BACKTRACE_TRIPLES "/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;492;add_test;/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;0;")
add_test(Seed "SMHasher" "--test=Seed")
set_tests_properties(Seed PROPERTIES  _BACKTRACE_TRIPLES "/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;493;add_test;/var/lib/phoronix-test-suite/installed-tests/pts/smhasher-1.0.0/smhasher-git/CMakeLists.txt;0;")
