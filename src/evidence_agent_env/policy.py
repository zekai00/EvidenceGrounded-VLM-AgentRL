"""Qwen-VL policy wrapper for executable trajectory rollout."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .prompting import PromptConfig, build_messages_from_observation


class QwenVLSftPolicy:
    def __init__(
        self,
        model_path: str | Path,
        adapter_path: str | Path | None = None,
        *,
        load_in_4bit: bool = True,
        torch_dtype: str = "bf16",
        image_max_pixels: int = 262144,
        max_seq_length: int = 14336,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: str = "",
        prompt_config: PromptConfig | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.image_max_pixels = image_max_pixels
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.prompt_config = prompt_config or PromptConfig()
        self.processor, self.model = self._load_model_and_processor()
        self.model.eval()

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        raw_text = self.generate(obs)
        return {
            "raw_text": raw_text,
            "action": extract_json_object(raw_text),
        }

    def generate(self, obs: dict[str, Any]) -> str:
        from qwen_vl_utils import process_vision_info

        messages = build_messages_from_observation(obs, self.prompt_config)
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        device = infer_input_device(self.model)
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        do_sample = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        with torch.no_grad():
            generated = self.model.generate(**inputs, **generation_kwargs)
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated[:, input_len:]
        return self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.model_path,
            "adapter": self.adapter_path,
            "load_in_4bit": self.load_in_4bit,
            "torch_dtype": self.torch_dtype,
            "image_max_pixels": self.image_max_pixels,
            "max_seq_length": self.max_seq_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "prompt_config": asdict(self.prompt_config),
        }

    def _load_model_and_processor(self) -> tuple[Any, Any]:
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        disable_autoawq_dispatch()
        dtype = parse_torch_dtype(self.torch_dtype)
        processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.image_max_pixels:
            processor_kwargs["max_pixels"] = self.image_max_pixels
        processor = AutoProcessor.from_pretrained(self.model_path, **processor_kwargs)
        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        model_kwargs["torch_dtype"] = dtype if dtype != "auto" else "auto"
        if self.load_in_4bit:
            compute_dtype = torch.bfloat16 if dtype == "auto" else dtype
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        model = AutoModelForImageTextToText.from_pretrained(self.model_path, **model_kwargs)
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path, is_trainable=False)
        model.config.use_cache = True
        return processor, model


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    starts = [index for index, char in enumerate(cleaned) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(cleaned[start : index + 1])
                            if isinstance(obj, dict) and obj.get("action"):
                                return obj
                        except Exception:
                            break
    return salvage_truncated_chunk(cleaned)


def salvage_truncated_chunk(text: str) -> dict[str, Any] | None:
    """Recover a bounded write_claims_chunk from a truncated outer JSON string."""

    if '"action"' not in text or "write_claims_chunk" not in text or '"claims"' not in text:
        return None
    claims_pos = text.find('"claims"')
    list_start = text.find("[", claims_pos)
    if list_start < 0:
        return None
    claims = extract_complete_objects_from_list(text, list_start, limit=2)
    if not claims:
        partial_claim = extract_partial_claim(text[list_start + 1 :])
        if partial_claim:
            claims = [partial_claim]
    abstains: list[dict[str, Any]] = []
    abstains_pos = text.find('"abstains"', list_start)
    if abstains_pos >= 0:
        abstain_start = text.find("[", abstains_pos)
        if abstain_start >= 0:
            abstains = extract_complete_objects_from_list(text, abstain_start, limit=max(0, 2 - len(claims)))
    if not claims and not abstains:
        return None
    return {"action": "write_claims_chunk", "claims": claims, "abstains": abstains}


def extract_partial_claim(text: str) -> dict[str, Any] | None:
    field = read_json_value_after_key(text, "field")
    value = read_json_value_after_key(text, "value")
    evidence_ids = read_json_value_after_key(text, "evidence_ids")
    visual_bbox = read_json_value_after_key(text, "visual_bbox")
    confidence = read_json_value_after_key(text, "confidence")
    if not field or value is None:
        return None
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    if confidence is None:
        confidence = 0.75
    return {
        "field": field,
        "value": value,
        "evidence_ids": evidence_ids,
        "visual_bbox": visual_bbox,
        "confidence": confidence,
    }


def read_json_value_after_key(text: str, key: str) -> Any:
    match = re.search(r'"' + re.escape(key) + r'"\s*:', text)
    if not match:
        return None
    index = match.end()
    while index < len(text) and text[index].isspace():
        index += 1
    try:
        value, _ = json.JSONDecoder().raw_decode(text[index:])
        return value
    except Exception:
        return None


def extract_complete_objects_from_list(text: str, list_start: int, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    index = list_start + 1
    while index < len(text) and len(items) < limit:
        start = text.find("{", index)
        if start < 0:
            break
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(text)):
            char = text[end]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start : end + 1])
                        except Exception:
                            return items
                        if isinstance(obj, dict):
                            items.append(obj)
                        index = end + 1
                        break
        else:
            break
    return items


def parse_torch_dtype(name: str) -> Any:
    normalized = str(name or "auto").lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def disable_autoawq_dispatch() -> None:
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.awq as peft_lora_awq

        peft_import_utils.is_auto_awq_available.cache_clear()
        peft_import_utils.is_auto_awq_available = lambda: False
        peft_lora_awq.is_auto_awq_available = lambda: False
    except Exception:
        return


def infer_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return torch.device(device)
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
