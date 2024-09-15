import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed
from transformers import GPT2Config, GPT2Model
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import json
import logging
from datetime import datetime

# Setup logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"train_log_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(process)d] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="DDP Training Script")
    parser.add_argument('--start_phase', type=int, required=True, help='Starting phase number')
    parser.add_argument('--end_phase', type=int, required=True, help='Ending phase number')
    return parser.parse_args()

def setup_ddp(rank, world_size):
    print(f"[Rank {rank}] Setting up DDP environment with world size {world_size}")
    os.environ['MASTER_ADDR'] = '172.26.112.16'  # Replace with the master node's IP
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logging.info(f"[Rank {rank}] DDP setup complete with rank {rank} and world size {world_size}")

def load_model_config(config_path):
    print(f"Loading model configuration from {config_path}")
    with open(config_path, "r") as file:
        config = json.load(file)
    print("Model configuration loaded successfully")
    return config["model_config"]

def initialize_model(model_config):
    print("Initializing model with configuration:", model_config)
    gpt2_config = GPT2Config(**model_config)
    model = GPT2Model(gpt2_config)
    print("Model initialized successfully")
    return model

def print_collective_gpu_memory(rank):
    # Log GPU memory stats
    total_memory = torch.cuda.get_device_properties(rank).total_memory / (1024**3)  # GB
    reserved_memory = torch.cuda.memory_reserved(rank) / (1024**3)  # GB
    allocated_memory = torch.cuda.memory_allocated(rank) / (1024**3)  # GB
    free_memory = reserved_memory - allocated_memory

    logging.info(f"[Rank {rank}] Total GPU Memory: {total_memory:.2f} GB, "
                 f"Allocated: {allocated_memory:.2f} GB, Free: {free_memory:.2f} GB")
    
    # Aggregate GPU memory stats across all nodes
    total_memory_tensor = torch.tensor([total_memory], dtype=torch.float32).cuda()
    dist.all_reduce(total_memory_tensor, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        logging.info(f"[Rank 0] Collective Total GPU Memory: {total_memory_tensor.item():.2f} GB")

def load_datasets(config, start_phase, end_phase, rank):
    print(f"[Rank {rank}] Loading datasets for phases {start_phase} to {end_phase}")
    for phase_num in range(start_phase, end_phase + 1):
        phase_key = f"phase{phase_num}"
        phase_info = config["phases"].get(phase_key)
        if phase_info:
            dataset_name = phase_info["dataset_name"]
            try:
                print(f"[Rank {rank}] Attempting to load dataset '{dataset_name}' for Phase {phase_num}")
                dataset = load_dataset(dataset_name, split="train")
                logging.info(f"[Rank {rank}] Successfully loaded dataset '{dataset_name}' for Phase {phase_num}")
            except Exception as e:
                logging.error(f"[Rank {rank}] Failed to load dataset {dataset_name} for Phase {phase_num}: {e}")

def main():
    args = parse_arguments()
    start_phase = args.start_phase
    end_phase = args.end_phase
    
    # Assume rank and world size are provided by torchrun
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    print(f"Running main function on Rank {rank} with world size {world_size}")
    
    # Setup DDP environment
    setup_ddp(rank, world_size)

    # Load configuration file
    config_path = os.path.expanduser("~/rachana_ddp/config/config.json")
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at: {config_path}")
        return
    with open(config_path, "r") as file:
        config = json.load(file)

    print(f"[Rank {rank}] Configuration loaded from {config_path}")

    # Load model configuration and initialize model
    model_config = load_model_config(config_path)
    model = initialize_model(model_config).to(rank)
    model = DDP(model, device_ids=[rank])
    logging.info(f"[Rank {rank}] Model initialized and wrapped in DDP")

    # Print collective GPU memory across all nodes
    print_collective_gpu_memory(rank)

    # Load datasets from Hugging Face
    load_datasets(config, start_phase, end_phase, rank)

    # Cleanup DDP
    dist.barrier()
    dist.destroy_process_group()
    print(f"[Rank {rank}] DDP environment cleaned up")

if __name__ == "__main__":
    main()
