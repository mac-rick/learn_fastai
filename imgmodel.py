# CLICK ME
from fastai.vision.all import *

def is_cat(x):
    if not x:  # checks for empty string, None, etc.
        return False
    return x[0].isupper()

if __name__ == '__main__':
    path = untar_data(URLs.PETS)/'images'
    print(path)
    
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=is_cat, item_tfms=Resize(224), num_workers=0)
        
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("GPU not available, using CPU")
    learn.fine_tune(1)
    learn.show_results()

    uploader = SimpleNamespace(data = ['images/download.jpeg'])

    img = PILImage.create(uploader.data[0])
    is_cat,_,probs = learn.predict(img)
    print(f"Is this a cat?: {is_cat}.")
    print(f"Probability it's a cat: {probs[1].item():.6f}")