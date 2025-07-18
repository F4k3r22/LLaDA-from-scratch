from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
import random
import torch
import time
import logging

# Thanks to https://github.com/SpotDylan/llada-8b-fine-tune/blob/main/finetune_llada_alignment.py 
# many references to be able to do the SFT correctly were taken from here

MAX_SEQ_LEN = 4096

def setup_logging(log_file="training_log.log", console_level=logging.INFO, file_level=logging.DEBUG):
    """Set up logging with different levels for file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(min(console_level, file_level))  # Set to the more detailed level
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler - captures everything including debug messages
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - only shows info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging(log_file="sft_training_log.log", console_level=logging.INFO, file_level=logging.DEBUG)

def forward_process(input_ids, eps=1e-3):
    """
    Apply the forward process to add noise to the input.
    
    Args:
        input_ids: Input token IDs
        eps: Small epsilon to avoid division by zero
        
    Returns:
        noisy_batch: Input with noise applied
        masked_indices: Boolean tensor indicating which tokens are masked
        p_mask: Probability of masking for each token
    """
    start_time = time.time()
    b, l = input_ids.shape
    
    # Log input shape
    logger.debug(f"Forward process input shape: batch_size={b}, seq_length={l}")
    
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    
    # Check for NaN or inf values in p_mask
    if torch.isnan(p_mask).any() or torch.isinf(p_mask).any():
        logger.error(f"NaN or inf values detected in p_mask: {p_mask}")
    
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    
    # Log masking statistics
    mask_percentage = masked_indices.float().mean().item() * 100
    logger.debug(f"Masking {mask_percentage:.2f}% of tokens")
    
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    
    logger.debug(f"Forward process completed in {time.time() - start_time:.4f} seconds")
    return noisy_batch, masked_indices, p_mask

device = 'cuda'
model = AutoModel.from_pretrained('Fredtt3/LLaDA-100M-Test', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('Fredtt3/LLaDA-100M-Test', trust_remote_code=True)