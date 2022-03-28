TLS Support
===========

Getting Started
---------------

### Building

To build with TLS support you'll need OpenSSL development libraries (e.g.
libssl-dev on Debian/Ubuntu).

Run `make BUILD_TLS=yes`.

### Tests

To run KeyDB test suite with TLS, you'll need TLS support for TCL (i.e.
`tcl-tls` package on Debian/Ubuntu).

1. Run `./utils/gen-test-certs.sh` to generate a root CA and a server
   certificate.

2. Run `./runtest --tls` or `./runtest-cluster --tls` to run KeyDB and KeyDB
   Cluster tests in TLS mode.

### Running manually

To manually run a Redis server with TLS mode (assuming `gen-test-certs.sh` was
invoked so sample certificates/keys are available):

    ./src/keydb-server --tls-port 6379 --port 0 \
        --tls-cert-file ./tests/tls/client.crt \
        --tls-key-file ./tests/tls/client.key \
        --tls-ca-cert-file ./tests/tls/ca.crt

To connect to this Redis server with `keydb-cli`:

    ./src/keydb-cli --tls \
        --cert ./tests/tls/keydb.crt \
        --key ./tests/tls/keydb.key \
        --cacert ./tests/tls/ca.crt

This will disable TCP and enable TLS on port 6379. It's also possible to have
both TCP and TLS available, but you'll need to assign different ports.

To make a Replica connect to the master using TLS, use `--tls-replication yes`,
and to make KeyDB Cluster use TLS across nodes use `--tls-cluster yes`.

Connections
-----------

All socket operations now go through a connection abstraction layer that hides
I/O and read/write event handling from the caller.

Note that unlike Redis, KeyDB fully supports multithreading of TLS connections.

To-Do List
----------

- [ ] keydb-benchmark support. The current implementation is a mix of using
  hiredis for parsing and basic networking (establishing connections), but
  directly manipulating sockets for most actions. This will need to be cleaned
  up for proper TLS support. The best approach is probably to migrate to hiredis
  async mode.
- [ ] keydb-cli `--slave` and `--rdb` support.

Multi-port
----------

Consider the implications of allowing TLS to be configured on a separate port,
making KeyDB listening on multiple ports:

1. Startup banner port notification
2. Proctitle
3. How slaves announce themselves
4. Cluster bus port calculation
