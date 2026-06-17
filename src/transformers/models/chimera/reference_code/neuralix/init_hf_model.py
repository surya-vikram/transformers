import sys
import os
import torch
import shutil

from configuration_neuralix import NeuralixConfig
from modeling_neuralix import NeuralixForCausalLM

def main():
    print("Initializing NeuralixConfig...")
    config = NeuralixConfig()
    
    # We must register the auto_map so that from_pretrained knows to look for these custom files
    config.auto_map = {
        "AutoConfig": "configuration_neuralix.NeuralixConfig",
        "AutoModelForCausalLM": "modeling_neuralix.NeuralixForCausalLM"
    }
    
    print("Instantiating NeuralixForCausalLM with random weights...")
    print("NOTE: Generating 11 Billion parameters requires ~22GB of RAM. Make sure you run this on a capable node.")
    
    # Initialize natively in bfloat16 to save memory
    model = NeuralixForCausalLM(config).to(torch.bfloat16)
    
    save_dir = "hf_neuralix_model"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving model to {save_dir}/...")
    model.save_pretrained(save_dir, safe_serialization=True)
    
    # Crucial step: Copy the custom python files into the save directory
    # so that `trust_remote_code=True` can find them!
    shutil.copy("configuration_neuralix.py", os.path.join(save_dir, "configuration_neuralix.py"))
    shutil.copy("modeling_neuralix.py", os.path.join(save_dir, "modeling_neuralix.py"))
    
    print(f"Neuralix-10B successfully saved! You can now load it via AutoModelForCausalLM.from_pretrained('{save_dir}', trust_remote_code=True)")

if __name__ == "__main__":
    main()
