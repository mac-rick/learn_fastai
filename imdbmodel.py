from fastai.vision.all import *
from fastai.text.all import TextDataLoaders, text_classifier_learner, AWD_LSTM, accuracy

if __name__ == '__main__':   
    dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
   
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("GPU not available, using CPU")
        
    print("Starting fine-tuning")
    learn.fine_tune(4, 1e-2)
    print("Fine-tuning complete")
    learn.show_results()
    print("Showing results")
    learn.predict("I really liked that movie!")