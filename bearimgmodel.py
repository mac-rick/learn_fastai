# CLICK ME
from fastai.vision.all import *
from fastdownload import download_url

import requests

def search_images_rapidapi(term, n=20):
    url = "https://real-time-image-search.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": "THE_API_KEY",
        "X-RapidAPI-Host": "real-time-image-search.p.rapidapi.com"
    }
    params = {"query": term, "limit": n, "size": "any", "color": "any", "type": "any"}
    r = requests.get(url, headers=headers, params=params).json()
    
    # Debugging: see what came back
    # print(r)
    
    # Adjust depending on what the API returns
    results = r.get("data", [])
    return [item.get("url") for item in results if item.get("url")]

class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])

if __name__ == '__main__':
    # ims = search_images_rapidapi("grizzly bear", n=1)
    # print(len(ims), ims[:3])

    # if ims:
    #     os.makedirs("images", exist_ok=True)
    #     dest = "images/grizzly.jpg"
    #     download_url(ims[0], dest)
    #     im = Image.open(dest)
    #     im.to_thumb(128, 128)
    # else:
    #     print("⚠️ No images found")

    path = Path("images")
    bear_types = ["grizzly", "black", "teddy"]

    # for o in bear_types:
    #     results = search_images_rapidapi(f"{o} bear", n=30)
    #     if results:
    #         dest = path / o
    #         dest.mkdir(parents=True, exist_ok=True)

    #         # find max index among existing files
    #         existing_files = list(dest.glob("*.jpg"))
    #         if existing_files:
    #             existing_numbers = []
    #             for f in existing_files:
    #                 try:
    #                     num = int(f.stem.split("_")[-1])
    #                     existing_numbers.append(num)
    #                 except ValueError:
    #                     continue
    #             start_idx = max(existing_numbers) + 1 if existing_numbers else 0
    #         else:
    #             start_idx = 0

    #         # save images with new numbers
    #         for i, url in enumerate(results, start=start_idx):
    #             fname = dest / f"{o}_{i}.jpg"
    #             try:
    #                 download_url(url, fname)
    #                 print(f"✅ saved {fname}")
    #             except Exception as e:
    #                 print(f"⚠️ failed to download {url} → {e}")
    #     else:
    #         print(f"⚠️ No images found for {o} bear")

    fns = get_image_files(path)
    # print(fns)

    bears = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128))

    # dls = bears.dataloaders(path)
    # dls.valid.show_batch(max_n=4, nrows=1)

    # bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
    # dls = bears.dataloaders(path)
    # dls.valid.show_batch(max_n=4, nrows=1)

    # bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
    # dls = bears.dataloaders(path)
    # print("Using Pad with zeros - validation batch info:")
    # batch = dls.valid.one_batch()
    # print(f"Batch shape: {batch[0].shape}")
    # print(f"Labels: {batch[1]}")
    # dls.valid.show_batch(max_n=4, nrows=1)
    # plt.savefig('images/pad_batch.png')
    # print("Pad batch saved as images/pad_batch.png")

    # bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
    # dls = bears.dataloaders(path)
    # # For terminal - show info about the batch instead of visual display
    # batch = dls.train.one_batch()
    # print(f"Batch shape: {batch[0].shape}")
    # print(f"Labels: {batch[1]}")
    # print(f"Label names: {dls.vocab}")
    
    # # Save a sample batch as image file to view
    # dls.train.show_batch(max_n=4, nrows=1, unique=True)
    # plt.savefig('images/sample_batch.png')
    # print("Sample batch saved as images/sample_batch.png")

    # # Define different transform strategies
    # resize_strategies = [
    #     ("Default Resize", Resize(128)),
    #     ("Squish", Resize(128, ResizeMethod.Squish)),
    #     ("Pad (zeros)", Resize(128, ResizeMethod.Pad, pad_mode='zeros')),
    #     ("RandomResizedCrop", RandomResizedCrop(128, min_scale=0.3))
    # ]

    # # Create figure with 2 rows × 2 cols
    # fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # for ax, (title, tfm) in zip(axs.flatten(), resize_strategies):
    #     dls = bears.new(item_tfms=tfm).dataloaders(path)
    #     dls.valid.show_batch(max_n=4, nrows=1, ax=ax)
    #     ax.set_title(title)

    # plt.tight_layout()
    # plt.savefig("images/resize_comparison.png")
    # plt.show()

    # bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
    # dls = bears.dataloaders(path)
    # dls.train.show_batch(max_n=8, nrows=2, unique=True)
    # plt.savefig('images/RandomResizedCrop_batch.png')
    # print("Sample batch saved as images/RandomResizedCrop_batch.png")

    bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
    dls = bears.dataloaders(path, num_workers=0)

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)
    
    # Show final results in terminal
    print("\nTraining completed!")
    print(f"Final validation error rate: {learn.validate()[1]:.4f}")
    
    # Show some predictions
    print("\nSample predictions:")
    # Get a single image from validation set
    test_files = get_image_files(path)
    if test_files:
        test_img = PILImage.create(test_files[0])
        pred_class, pred_idx, probs = learn.predict(test_img)
        print(f"Image: {test_files[0].name}")
        print(f"Predicted: {pred_class}, Confidence: {probs.max():.4f}")
    else:
        print("No test images found")
    
    # Save confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    plt.savefig('outcome/confusion_matrix.png')
    print("Confusion matrix saved as outcome/confusion_matrix.png")

    interp.plot_top_losses(5, nrows=1)
    plt.savefig('outcome/top_losses.png')
    print("Top losses saved as outcome/top_losses.png")
    
    # Get indices and details of top loss images
    top_losses_vals, top_losses_idx = interp.top_losses(5)
    print(f"\nTop 5 loss values: {top_losses_vals}")
    print(f"Top 5 loss indices: {top_losses_idx}")
    
    # Get more detailed info about each top loss image
    for i, idx in enumerate(top_losses_idx):
        idx = int(idx)  # Convert tensor to int
        actual_label = dls.valid.dataset.items[idx].parent.name
        pred_class, pred_idx, probs = learn.predict(dls.valid.dataset[idx][0])
        loss_val = float(top_losses_vals[i])  # Convert tensor to float
        print(f"Rank {i+1}: Index {idx}, Actual: {actual_label}, Predicted: {pred_class}, Loss: {loss_val:.4f}, Max Prob: {probs.max():.4f}")


    # So, for instance, to delete (unlink) all images
    # selected for deletion, we would run this:
    # for idx in cleaner.delete(): cleaner.fns[idx].unlink()

    # To move images for which we’ve selected a different category, we would run this:
    # for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)

    # ImageClassifierCleaner requires Jupyter widgets - not available in terminal
    # For terminal use, the confusion matrix and top losses above provide good insights

    learn.export()

    path = Path()
    pkl_files = path.ls(file_exts='.pkl')
    print(f"\nExported model files (.pkl): {pkl_files}")
    
    # Load the exported model and test prediction
    learn_inf = load_learner('export.pkl')
    
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
    