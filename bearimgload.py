# CLICK ME
from fastai.vision.all import *
from fastdownload import download_url

from matplotlib import widgets
import requests
from IPython.display import display

if __name__ == '__main__':

    path = Path()
    pkl_files = path.ls(file_exts='.pkl')
    print(f"\nExported model files (.pkl): {pkl_files}")

    learn_inf = load_learner(path/'export.pkl')
    learn_inf.predict('test/grizzly.jpg')

    # Test with an image (replace with actual test image path)
    test_image_path = 'test/grizzly.jpg'
    if Path(test_image_path).exists():
        pred_class, pred_idx, probs = learn_inf.predict(test_image_path)
        print(f"\nPrediction for {test_image_path}:")
        print(f"Predicted class: {pred_class}")
        print(f"Confidence: {probs.max():.4f}")
        print(f"All probabilities: {dict(zip(learn_inf.dls.vocab, probs))}")
    else:
        print(f"\nTest image not found: {test_image_path}")
        # Use any available image instead
        available_images = get_image_files('images')
        if available_images:
            test_img = available_images[0]
            pred_class, pred_idx, probs = learn_inf.predict(test_img)
            print(f"\nPrediction for {test_img}:")
            print(f"Predicted class: {pred_class}")
            print(f"Confidence: {probs.max():.4f}")
            print(f"All probabilities: {dict(zip(learn_inf.dls.vocab, probs))}")
    

    btn_run = widgets.Button(description='Classify')
    display(btn_run)