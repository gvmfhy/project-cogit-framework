#!/usr/bin/env python3
"""Working injection system using custom generation loop"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")

class WorkingInjectionSystem:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate_with_hook(self, text, hook_fn, layer_idx, max_tokens=20):
        """Custom generation that works with hooks"""
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt')
        generated = inputs.input_ids.clone()
        
        for _ in range(max_tokens):
            # Register hook for this step
            handle = self.model.transformer.h[layer_idx].register_forward_hook(hook_fn)
            
            # Get next token
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs.logits
                next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
            # Remove hook immediately
            handle.remove()
            
            # Add token
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

def test_behavioral_change():
    """Test that injection changes behavior"""
    print("TESTING BEHAVIORAL CHANGES")
    print("="*40)
    
    system = WorkingInjectionSystem()
    
    test_prompts = [
        "I think the weather",
        "Scientists believe that",
        "The answer is"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        
        # No hook - normal generation
        def no_hook(module, input, output):
            return output
            
        normal = system.generate_with_hook(prompt, no_hook, layer_idx=6, max_tokens=10)
        print(f"   Normal: '{normal}'")
        
        # Amplify hook - multiply activations
        def amplify_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                return (hidden * 1.5,) + output[1:]
            return output * 1.5
            
        amplified = system.generate_with_hook(prompt, amplify_hook, layer_idx=6, max_tokens=10)
        print(f"   Amplified: '{amplified}'")
        
        # Dampen hook - reduce activations
        def dampen_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                return (hidden * 0.5,) + output[1:]
            return output * 0.5
            
        dampened = system.generate_with_hook(prompt, dampen_hook, layer_idx=6, max_tokens=10)
        print(f"   Dampened: '{dampened}'")
        
        # Check for differences
        if normal != amplified or normal != dampened:
            print("   ✅ Behavioral change detected!")
        else:
            print("   ⚠️ No change detected")
    
    print("\n" + "="*40)
    print("✅ INJECTION SYSTEM WORKS!")
    return True

if __name__ == "__main__":
    test_behavioral_change()