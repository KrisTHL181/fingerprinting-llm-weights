#!/usr/bin/env python3
import torch, copy, traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from align import Aligner

Q = "Qwen/Qwen3.5-0.8B"
def main():
    a = Aligner(Q, layer=8, device="cuda", dtype=torch.bfloat16)
    print("base first param device:", next(a.model.parameters()).device, flush=True)
    cfg = LoraConfig(r=16, lora_alpha=32,
                     target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                     task_type="CAUSAL_LM")
    m = get_peft_model(copy.deepcopy(a.model), cfg)
    print("peft first param device:", next(m.parameters()).device, flush=True)
    m = m.to(device="cuda", dtype=torch.bfloat16)
    print("after .to(cuda,bf16) first param device:", next(m.parameters()).device, flush=True)
    ids = a.tok("Hello world, this is a test.", add_special_tokens=False)["input_ids"]
    inp = torch.tensor([ids], dtype=torch.long, device="cuda")
    print("inp device:", inp.device, flush=True)
    m.train()
    out = m(input_ids=inp).logits
    ce = torch.nn.functional.cross_entropy(out[0, :-1].float().reshape(-1, out.size(-1)), inp[0, 1:].reshape(-1))
    ce.backward()
    print("FORWARD+BACKWARD OK, loss", ce.item(), "logits", tuple(out.shape), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FAILED:", repr(e), flush=True)
        traceback.print_exc()
