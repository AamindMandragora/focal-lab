# Import Dafny runtime library for Dafny-to-Python interoperability
import _dafny as _dafny
# Import System module used by Dafny generated code
import System_ as System_

# Import Hugging Face components for model and tokenizer handling
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
# Import configuration for model quantization (4-bit loading)
from transformers import BitsAndBytesConfig
# Import Lark parser for grammar-based constrained decoding
from lark import Lark
# Import exceptions to handle parsing errors during prefix validation
from lark.exceptions import UnexpectedInput, UnexpectedEOF
# Import Tensor class from PyTorch for type hinting
from torch import Tensor
# Import PyTorch library for tensor operations and device management
import torch
# Import standard typing utilities for type hints
from typing import Optional, List, Union

# Define the specific pre-trained model identifier to load
MODEL_NAME = "Qwen/Qwen2-Math-72B"
# Define the filename of the grammar file used for parsing
GRAMMAR_FILE = "./grammars/gsm.lark"
# Specify the device to run the model on (GPU)
DEVICE = "cuda"
# Set the maximum number of new tokens to generate in one step
MAX_NEW_TOKENS = 3

# Define a wrapper class to manage the LLM and its generation process
class LocalLLM:
    """
    A lightweight wrapper around HuggingFace model and tokenizer to support generation 
    with dynamic LogitsProcessor injection.
    """
    def __init__(self, model_name, device, quantize=True):
        # Initialize the LocalLLM instance with model name, device, and quantization flag
        
        # Load the tokenizer associated with the model, allowing remote code execution
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Initialize quantization configuration variable as None
        quantization_config = None
        # Check if quantization is enabled
        if quantize:
            # Create a BitsAndBytesConfig object for 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                # Enable loading the model in 4-bit precision to save memory
                load_in_4bit=True,
                # Set the computation data type to float16 for speed and memory efficiency
                bnb_4bit_compute_dtype=torch.float16,
                # Enable double quantization to further reduce memory usage
                bnb_4bit_use_double_quant=True,
            )
        
        # Load the pre-trained causal language model
        self.model = AutoModelForCausalLM.from_pretrained(
            # Pass the model name
            model_name,
            # Automatically map layers to GPU if cuda is selected, else no map
            device_map="auto" if device == "cuda" else None,
            # Apply the quantization configuration defined above
            quantization_config=quantization_config,
            # Allow execution of custom code from the model repository
            trust_remote_code=True,
            # Set torch data type: float16 for GPU, float32 for CPU
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Check if running on CPU (or non-cuda) and quantization is disabled
        if device != "cuda" and not quantize:
             # Explicitly move the model to the specified device (e.g., "cpu")
             self.model.to(device)

    def generate(self, prompt: str, max_new_tokens: int = 10, logits_processor: Optional[LogitsProcessor] = None):
        # Define generation method with optional logits processor
        
        # Tokenize the prompt and move tensors to the model's device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Create an empty list to hold logits processors
        logits_processors = LogitsProcessorList()
        # Check if a custom logits processor was provided
        if logits_processor:
            # Add the provided logits processor to the list
            logits_processors.append(logits_processor)
        
        # Disable gradient calculation to save memory and computation during inference
        with torch.no_grad():
            # Generate text using the model
            outputs = self.model.generate(
                # Unpack input tensors (input_ids, attention_mask)
                **inputs,
                # Limit the number of new tokens generated
                max_new_tokens=max_new_tokens,
                # Specify the padding token ID
                pad_token_id=self.tokenizer.pad_token_id,
                # Apply the list of logits processors (including constraints)
                logits_processor=logits_processors,
                # Disable sampling (use greedy decoding) for deterministic output
                do_sample=False,
                # Enable KV cache to speed up generation
                use_cache=True
            )
        
        # Return only the newly generated tokens, slicing off the prompt tokens
        return outputs[0][inputs.input_ids.shape[1]:]

# Initialize components

# Print a message indicating the model is loading
print(f"Loading model {MODEL_NAME}...")
# Instantiate the LocalLLM with the specified configuration
llm = LocalLLM(MODEL_NAME, DEVICE, quantize=True)
# Expose the tokenizer for external use
tokenizer = llm.tokenizer

# Open the grammar file in read mode
with open(GRAMMAR_FILE, "r") as f:
    # Read the entire content of the grammar file
    grammar_content = f.read()

# Initialize the Lark parser with the grammar content starting at the 'start' rule
constrained = Lark(grammar_content, start="start")

def Parser__ValidPrefix(prefix: str):
    # Define a function to check if a string prefix is valid according to the grammar
    """Checks if prefix is valid under the current grammar."""
    try:
        # Attempt to parse the prefix using the Lark parser
        constrained.parse(prefix, start="start")
        # Return True if parsing succeeds (complete valid sentence)
        return True
    except UnexpectedInput as e:
        # Catch UnexpectedInput exceptions (parsing failed)
        # Return True if the error is UnexpectedEOF (valid prefix but incomplete), False otherwise
        return isinstance(e, UnexpectedEOF)

def Parser__IsComplete(prefix: str):
    """Checks if prefix is complete under the current grammar."""
    try:
        constrained.parse(prefix, start="start")
        return True
    except UnexpectedInput:
        return False

def Parser__AllowedNext(prefix: str):
    """Return the set of expected terminals for the current prefix."""
    try:
        constrained.parse(prefix, start="start")
        return []
    except UnexpectedEOF as e:
        return list(set(e.expected))
    except UnexpectedInput:
        return []

def Generator__ChooseToken(prefix: str, allowed: list[str], pointer: int, is_constrained: bool = False, logits_processor: Optional[LogitsProcessor] = None):
    # Define function to generate and choose a token
    """
    Generates token based on prefix and properly detokenizes it.
    """
    # Extract the currently generated part of the string starting from pointer
    curr_gen = prefix[pointer:]
    # Check if the constrained marker "<<" is present in the current generation
    if "<<" in curr_gen:
        # Set constrained flag to True
        is_constrained = True
    else:
        # Set constrained flag to False
        is_constrained = False
    
    # Select the logits processor to use based on the constrained flag
    processor = logits_processor if is_constrained else None
    # The point of this line is that the logits processor should only be used
    # if constrained mode is active (like it is in the broader CRANE framework)
    
    # Generate tokens using the LLM instance
    tokens = llm.generate(prefix, max_new_tokens=MAX_NEW_TOKENS, logits_processor=processor)
    
    # Convert generated tensor to a list of IDs, handling batch dimension
    generated_ids = tokens[0].tolist() if tokens.dim() > 1 else tokens.tolist()
    
    # Convert the list of token IDs back into token strings
    decoded_tokens = llm.tokenizer.convert_ids_to_tokens(generated_ids)
    # Print the decoded tokens for debugging
    print(decoded_tokens)
    # Return the list of decoded tokens
    return decoded_tokens

if __name__ == "__main__":
    # Do nothing if run as main (placeholder)
    pass
