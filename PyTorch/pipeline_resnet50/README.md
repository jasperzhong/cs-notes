## Reproduce

Run `run.sh` on both machines. 
```sh
$ ./run.sh 
```

## Environment

See https://github.com/pytorch/pytorch/issues/64926 for more details. 

Basically, we have two machines. Each machine has two NICs. One is `eth0` and the other is `eth2`. The `eth2` NIC is connected to a 50 Gbps switch and thus we prefer to use it. 

Machine One: 
- `eth0`: 202.45.128.221
- `eth2`: 10.28.1.16

Machine Two:
- `eth0`: 202.45.128.222
- `eth2`: 10.28.1.17
