"""
Local answer generation with a small open instruction-tuned LLM.

Model: Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0). Chosen because it runs
locally on one GPU (~8 GB in bf16, no paid API), follows formatting
instructions such as citation formats well, and the "Instruct-2507" variant
answers directly without long reasoning traces. Greedy decoding keeps
answers reproducible.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation.context_builder import TOP_K_CONTEXT, build_messages, check_citations


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class LocalLLM:

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()

    @torch.inference_mode()
    def generate(self, messages: list[dict], max_new_tokens: int = 350) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()


def answer_query(query: str, trials: list[dict[str, Any]], llm: LocalLLM,
                 k: int = TOP_K_CONTEXT) -> dict[str, Any]:
    """Generate a grounded answer from the top-k trials and check its citations."""
    messages = build_messages(query, trials, k)
    start = time.perf_counter()
    answer = llm.generate(messages)
    return {
        "answer": answer,
        "latency_s": round(time.perf_counter() - start, 2),
        "context_nct_ids": [t["nct_id"] for t in trials[:k]],
        **check_citations(answer, trials, k),
    }
