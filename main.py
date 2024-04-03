
import random
from process.train import run_train
from process.inference import inference_instance, inference_semantic
from data.load_local import get_dataset, get_images
from PIL import Image

train_images, train_masks, valid_images, valid_masks = get_images("/Users/vika/Desktop/TablesSemantic")
train_data, val_data = get_dataset(train_images, train_masks, valid_images, valid_masks)

run_train('instance', train_data, val_data, 10, 10, lr=1e-4)

#index = random.randint(0, len(train_data)-1)
#image = train_data[index]["image"].convert("RGB")
image = Image.open('/Users/vika/Desktop/Screenshot 2024-04-03 at 10.18.04.png').convert("RGB")
inference_semantic(image)
