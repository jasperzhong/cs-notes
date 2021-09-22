import os
from datetime import timedelta

import torch
import torch.distributed.rpc as rpc


def main():
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    torch.distributed.init_process_group(
        'gloo', world_size=world_size,
        rank=rank, timeout=timedelta(seconds=10)
    )

    name = "worker{}".format(rank)
    rpc.init_rpc(name, rank=rank, world_size=world_size,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                     init_method="tcp://localhost:29401"
                 ))

    torch.distributed.barrier()



if __name__ == "__main__":
    main()
