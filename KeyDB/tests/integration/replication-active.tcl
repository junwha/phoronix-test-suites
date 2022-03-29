start_server {tags {"active-repl"} overrides {active-replica yes}} {
    set slave [srv 0 client]
    set slave_host [srv 0 host]
    set slave_port [srv 0 port]
    set slave_log [srv 0 stdout]
    set slave_pid [s process_id]

    start_server [list overrides [list active-replica yes replicaof [list $slave_host $slave_port]]] {
        set master [srv 0 client]
        set master_host [srv 0 host]
        set master_port [srv 0 port]
    	set master_pid [s process_id]

        # Use a short replication timeout on the slave, so that if there
        # are no bugs the timeout is triggered in a reasonable amount
        # of time.
        $slave config set repl-timeout 5
        $master config set repl-timeout 5

        # Start the replication process...
        $slave slaveof $master_host $master_port
        #note the master is a replica via the config (see start_server above)

        test {Active replicas report the correct role} {
            wait_for_condition 50 100 {
                [string match *active-replica* [$slave role]]
            } else {
                fail "Replica0 does not report the correct role"
            }
            wait_for_condition 50 100 {
                [string match *active-replica* [$master role]]
            } else {
                fail "Replica1 does not report the correct role"
            }
        }

        test {Active replicas propogate} {
            $master set testkey foo
            after 500
            wait_for_condition 50 500 {
                [string match *foo* [$slave get testkey]]
            } else {
                fail "replication failed to propogate"
            }

            $slave set testkey bar
            wait_for_condition 50 500 {
                [string match bar [$master get testkey]]
            } else {
                fail "replication failed to propogate in the other direction"
            }
        }

        test {Active replicas propogate binary} {
            $master set binkey "\u0000foo"
            wait_for_condition 50 500 {
                [string match *foo* [$slave get binkey]]
            } else {
                fail "replication failed to propogate binary data"
            }
        }

        test {Active replicas propogate transaction} {
            $master set testkey 0
            $master multi
            $master incr testkey
            $master incr testkey
            after 5000
            $master get testkey
            $master exec
            assert_equal 2 [$master get testkey]
            after 500
            wait_for_condition 50 500 {
                [string match "2" [$slave get testkey]]
            } else {
                fail "Transaction failed to replicate"
            }
            $master flushall
        }

        test {Replication of EXPIREMEMBER (set) command (Active)} {
            $master sadd testkey a b c d
            wait_for_condition 50 200 {
                [$master debug digest] eq [$slave debug digest]
            } else {
                fail "Failed to replicate set"
            }
            $master expiremember testkey a 1
            after 1000
            wait_for_condition 50 100 {
                [$master scard testkey] eq 3
            } else {
                fail "expiremember failed to work on master"
            }
            wait_for_condition 50 100 {
                [$slave scard testkey] eq 3
            } else {
                assert_equal [$slave scard testkey] 3
            }
            $master del testkey
        }

		test {Replication of EXPIREMEMBER (hash) command (Active)} {
            $master hset testkey a value
            $master hset testkey b value
            wait_for_condition 50 100 {
                [$master debug digest] eq [$slave debug digest]
            } else {
                fail "Failed to replicate set"
            }
            $master expiremember testkey a 1
            after 1000
            wait_for_condition 50 100 {
                [$master hlen testkey] eq 1
            } else {
                fail "expiremember failed to work on master"
            }
            wait_for_condition 50 100 {
                [$slave hlen testkey] eq 1
            } else {
                assert_equal [$slave hlen testkey] 1
            }
            $master del testkey
        }

        test {Replication of EXPIREMEMBER (zset) command (Active)} {
            $master zadd testkey 1 a
            $master zadd testkey 2 b
            wait_for_condition 50 100 {
                [$master debug digest] eq [$slave debug digest]
            } else {
                fail "Failed to replicate set"
            }
            $master expiremember testkey a 1
            after 1000
            wait_for_condition 50 100 {
                [$master zcard testkey] eq 1
            } else {
                fail "expiremember failed to work on master"
            }
            wait_for_condition 50 100 {
                [$slave zcard testkey] eq 1
            } else {
                assert_equal [$slave zcard testkey] 1
            }
        }

        test {keydb.cron replicates (Active) } {
            $master del testkey
	        $master keydb.cron testjob repeat 0 1000000 {redis.call("incr", "testkey")} 1 testkey
         	after 300
    	    assert_equal 1 [$master get testkey]
	        assert_equal 1 [$master exists testjob]
			
			wait_for_condition 50 100 {
				[$master debug digest] eq [$slave debug digest]
			} else {
                fail "KEYDB.CRON failed to replicate"
            }
        	$master del testjob
            $master del testkey
            wait_for_condition 50 1000 {
                [$master debug digest] eq [$slave debug digest]
            } else {
                fail "cron delete failed to propogate"
            }
    	}

        test {Active replicas WAIT} {
            # Test that wait succeeds since replicas should be syncronized
            $master set testkey foo
            $slave set testkey2 test
            assert_equal {1} [$master wait 1 1000] { "value should propogate
                within 1 second" }
            assert_equal {1} [$slave wait 1 1000] { "value should propogate
                within 1 second" }

            # Now setup a situation where wait should fail
            exec kill -SIGSTOP $slave_pid
            $master set testkey fee
            assert_equal {0} [$master wait 1 1000] { "slave shouldn't be
                synchronized since its stopped" }
        }
        # Resume the replica we paused in the prior test
        exec kill -SIGCONT $slave_pid

        test {Active replica expire propogates} {
            $master set testkey1 foo
            wait_for_condition 50 1000 {
                [string match *foo* [$slave get testkey1]]
            } else {
                fail "Replication failed to propogate"
            }
            $master pexpire testkey1 200
            after 1000
            assert_equal {0} [$master del testkey1] {"master expired"}
            assert_equal {0} [$slave del testkey1]  {"slave expired"}

            $slave set testkey1 foo px 200
            after 1000
            assert_equal {0} [$master del testkey1]
            assert_equal {0} [$slave del testkey1]
        }

    test {Active replica expire propogates when source is down} {
        $slave flushall
        $slave set testkey2 foo
        $slave set testkey1 foo
        wait_for_condition 50 1000 {
            [string match *foo* [$master get testkey1]]
        } else {
            fail "Replication failed to propogate"
        }
        $slave expire testkey1 2
        assert_equal {1} [$slave wait 1 500] { "value should propogate
                    within 0.5 seconds" }
        exec kill -SIGSTOP $slave_pid
        after 3000
        # Ensure testkey1 is gone.  Note, we can't do this directly as the normal commands lie to us
        # about what is actually in the dict.  The only way to know is with a count from info
        assert_equal {1} [expr [string first {keys=1} [$master info keyspace]] >= 0]  {"slave expired"}
    }
    
    exec kill -SIGCONT $slave_pid

    test {Active replica merge works when reconnecting} {
        $slave flushall
        $slave set testkey foo
        wait_for_condition 50 1000 {
            [string match *foo* [$master get testkey]]
        } else {
            fail "Replication failed to propogate"
        }
        $slave replicaof no one
        $master replicaof no one
        after 100
        $master set testkey baz
        after 200
        $slave set testkey bar
        after 100
        $slave replicaof $master_host $master_port
        after 1000
        $master replicaof $slave_host $slave_port
        after 1000

        assert_equal {bar} [$slave get testkey]  {replica is correct}
        assert_equal {bar} [$master get testkey] {master is correct}
    }

    test {Active replica merge works with client blocked} {
        $slave flushall
        $slave replicaof no one
        $master replicaof no one
        after 100
        set rd [redis_deferring_client]
        $rd blpop testlist 0
        $slave lpush testlist foo
        
        #OK Now reconnect
        $slave replicaof $master_host $master_port
        $master replicaof $slave_host $slave_port
        after 1000

        $rd read
    } {testlist foo}

    test {Active replica different databases} {
        $master select 3
        $master set testkey abcd
        $master select 2
        $master del testkey
        $slave select 3
        wait_for_condition 50 1000 {
            [string match abcd [$slave get testkey]]
        } else {
            fail "Replication failed to propogate DB 3"
        }
    }
}
}

start_server {tags {"active-repl"} overrides {active-replica yes}} {
    set slave [srv 0 client]
    set slave_host [srv 0 host]
    set slave_port [srv 0 port]
    start_server {tags {"active-repl"} overrides { active-replica yes}} {
        r set testkeyB bar
        test {Active Replica Merges Database On Sync} {
            $slave set testkeyA foo
            r replicaof $slave_host $slave_port
	    wait_for_condition 50 1000 {
                [string match *active-replica* [r role]]
            } else {
                fail "Replica did not connect"
            }
	    wait_for_condition 50 1000 {
		[string match "2" [r dbsize]]
            } else {
                fail "key did not propogate"
            }
	}
    }
}

start_server {tags {"active-repl"} overrides {active-replica yes}} {
    set master [srv 0 client]
    set master_host [srv 0 host]
    set master_port [srv 0 port]
    test {REPLICAOF no one in config properly clears master list} {
        start_server [list overrides [list "replicaof" "$master_host $master_port" "replicaof" "no one" "replicaof" "$master_host $master_port"]] {
            wait_for_condition 50 100 {
                [string match {*role:slave*} [[srv 0 client] info replication]] &&
                [string match "*master_host:$master_host*" [[srv 0 client] info replication]] &&
                [string match "*master_port:$master_port*" [[srv 0 client] info replication]]
            } else {
                fail "Replica did not properly connect to master"
            }
        }
	}
}
