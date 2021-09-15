import torch

_GLOBAL_ARGS = None

def initialize_global_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args

def is_pipeline_last_stage():
    return get_pipeline_model_parallel_rank() == \
            get_pipeline_model_parallel_world_size() - 1

def is_pipeline_first_stage():
    return get_pipeline_model_parallel_rank() == 0

def get_pipeline_model_parallel_world_size():
    return torch.distributed.get_world_size()

def get_pipeline_model_parallel_rank():
    return torch.distributed.get_rank()

def get_pipeline_model_parallel_next_rank():
    return (get_pipeline_model_parallel_rank() + 1) % \
            get_pipeline_model_parallel_world_size()

def get_pipeline_model_parallel_prev_rank():
    return (get_pipeline_model_parallel_rank() - 1) % \
            get_pipeline_model_parallel_world_size()

def get_num_microbatches():
    return _GLOBAL_ARGS.global_batch_size // _GLOBAL_ARGS.micro_batch_size

def forward_step(data_iterator, model, input_tensor, loss_func):
    if is_pipeline_first_stage() or is_pipeline_last_stage():
        data = next(data_iterator)
        images, labels = data

    if is_pipeline_first_stage():
        assert input_tensor is None
        input_tensor = images

    output_tensor = model(input_tensor)

    if is_pipeline_last_stage():
        loss = loss_func(output_tensor, labels)
        output_tensor = loss / get_num_microbatches()

    return output_tensor

def send_forward(output_tensor):
    if not is_pipeline_last_stage():
        torch.distributed.isend(output_tensor, get_pipeline_model_parallel_next_rank())

def pipedream_flush_schedule(data_iterator, model, optimizer):
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = get_pipeline_model_parallel_world_size() - \
            get_pipeline_model_parallel_rank() - 1
    num_microbatches_remaining = \
            num_microbatches - num_warmup_microbatches

    input_tensors = []
    output_tensor = []

    # run warmup forward passes
    for i in range(num_warmup_microbatches):
        pass
