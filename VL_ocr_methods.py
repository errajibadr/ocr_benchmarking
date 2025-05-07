## git+https://github.com/illuin-tech/colpali and torchvision

from pathlib import Path
from typing import cast

import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="mps",  # "cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = cast(ColQwen2Processor, ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0"))

# Your inputs
images = [Image.open(str(image)) for image in Path("dataset/sample/images").glob("*.png")]
# images = [Image.open("dataset/sample/images/82200067_0069.png")]
images = images[:3]
queries = [
    "guidelines for advertising",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
print(scores)
