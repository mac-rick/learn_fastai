from fastai.vision.all import *

if __name__ == '__main__':
    path = untar_data(URLs.CAMVID_TINY)
    print(path)
    
    dls = SegmentationDataLoaders.from_label_func(
        path, bs=8, fnames = get_image_files(path/"images"),
        label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
        codes = np.loadtxt(path/'codes.txt', dtype=str),
        num_workers=0
    )
    
    learn = unet_learner(dls, resnet34)      
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("GPU not available, using CPU")
        
    print("Starting fine-tuning")
    learn.fine_tune(8)
    print("Fine-tuning complete")
    learn.show_results()
    print("Showing results")
    learn.show_results(max_n=6, figsize=(7,8))