# deeplearning2026

The project includes:
- Open-vocabulary detection
- Point-based segmentation
- Visual question answering (VQA)

## Foundation Models

- **Grounding DINO** for open-vocabulary detection
- **SAM2** for point-based segmentation
- **Qwen3-VL** for visual question answering

## Installation Guide
It is recommended to use Python 3.10 or later.
Install dependencies with:
pip install -r requirements.txt

## Brief explanation of each codes

## 1. open_vocab_det.py
Function
Performs open-vocabulary object detection / segmentation-related processing using Grounding DINO and SAM2.

Input:
Image file
Text queries such as person, book, bag

Output:
Bounding boxes
Visualization image
Optional JSON file
Optional masks

python code/open_vocab_det.py --image_path assets/input_examples/test1.png --text_queries person,book --save_json --save_masks

## 2. seg.py
Function
Performs point-based segmentation using SAM2.

Input:
Image file
User interaction with positive and negative points

Output:
Segmentation mask
Overlay image

python code/seg.py --image_path assets/input_examples/test1.png

Left click: positive point
Right click: negative point
Enter / Space: run segmentation
u: undo
c: clear
s: save
q: quit

## 3. VQA.py

Function
Performs visual question answering using Qwen3-VL.

Input:
Image file
Natural language question

Output:
Model-generated text answer
Optional JSON result

python code/VQA.py --image_path assets/input_examples/test1.png --question "What objects are visible in this image?" --out_json assets/result_examples/vqa_result.json

## Example Results

Please see the assets/result_examples/ directory for sample outputs.

## Notes
1.Different GPUs may produce different runtime speed.
2.Smaller checkpoints may be more stable on local GPUs.
3.For VQA, English questions are recommended for more stable outputs.
