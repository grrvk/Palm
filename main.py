# strange stupid machine
import random
from process.train import run
from process.inference import inf
from data.load_local import get_dataset, get_images

train_images, train_masks, valid_images, valid_masks = get_images("/Users/vika/Desktop/TablesSemantic")
train_data, val_data = get_dataset(train_images, train_masks, valid_images, valid_masks)

run(train_data, val_data, 1, 4, lr=5e-5)


'''index = random.randint(0, len(train_data)-1)
image = train_data[index]["image"].convert("RGB")
inf(image)'''
