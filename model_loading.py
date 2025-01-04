from transformers import (
    AutoProcessor,
    Mask2FormerForUniversalSegmentation
)

checkpoint = "facebook/mask2former-swin-large-coco-instance"
processor = AutoProcessor.from_pretrained(checkpoint)
model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)

print("Loaded Mask2Former model + processor!")
