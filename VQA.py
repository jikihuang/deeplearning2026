import os
import gc
import json
import argparse

import torch


def get_best_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def move_batch_to_device(batch, device, dtype=None):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.is_floating_point() and dtype is not None:
                moved[k] = v.to(device=device, dtype=dtype)
            else:
                moved[k] = v.to(device=device)
        else:
            moved[k] = v
    return moved


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_qwen3_vl(image_path: str, question: str, device: str, max_new_tokens: int) -> str:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    dtype = get_best_dtype()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_batch_to_device(inputs, device=device, dtype=dtype)
    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_len:]

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    del model, processor, inputs, generated_ids
    cleanup()
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("=" * 80)
    print("[Running] qwen3vl")

    try:
        answer = run_qwen3_vl(
            image_path=image_path,
            question=args.question,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        result = {
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "image_path": image_path,
            "question": args.question,
            "answer": answer,
        }
        print(f"[Answer] {answer}")
    except Exception as e:
        result = {
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "image_path": image_path,
            "question": args.question,
            "answer": None,
            "error": repr(e),
        }
        print(f"[ERROR] {repr(e)}")
    finally:
        cleanup()

    print("=" * 80)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out_json:
        out_path = os.path.abspath(args.out_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()