import torch

torch.cuda.empty_cache()
print('Total memory:',torch.cuda.get_device_properties(0).total_memory)
print('Researved:',torch.cuda.memory_reserved(0))
print('Allocated:',torch.cuda.memory_allocated(0))
print('Free:',torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0))
print('\n\n')
