from fastai.vision.all import *
from fastai.tabular.all import TabularDataLoaders, tabular_learner, Categorify, FillMissing, Normalize

if __name__ == '__main__':
    path = untar_data(URLs.ADULT_SAMPLE)
    print(path)
    
    dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
        cat_names = ['workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race'],
        cont_names = ['age', 'fnlwgt', 'education-num'],
        procs = [Categorify, FillMissing, Normalize])
    
    learn = tabular_learner(dls, metrics=accuracy)

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("GPU not available, using CPU")
        
    print("Starting tuning")
    learn.fit_one_cycle(3)
    print("tuning complete")   

    print("Showing validation predictions:")
    preds, targets = learn.get_preds()
    
    # Show first 10 predictions vs actual
    for i in range(min(10, len(preds))):
        pred_class = ">=50k" if preds[i][1] > 0.5 else "<50k"
        actual_class = ">=50k" if targets[i] == 1 else "<50k"
        confidence = preds[i][1].item()
        print(f"Sample {i+1}: Predicted={pred_class} (conf: {confidence:.3f}), Actual={actual_class}")
    
    # Show overall metrics
    from sklearn.metrics import classification_report
    pred_labels = (preds[:, 1] > 0.5).int()
    print("\nClassification Report:")
    print(classification_report(targets.cpu(), pred_labels.cpu(), target_names=["<50k", ">=50k"]))
