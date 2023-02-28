# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import json

os.environ["USE_TORCH"] = "1"

import multiprocessing as mp
import time
import pandas as pd
import fastwer

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms import Normalize
from tqdm import tqdm

from doctr import datasets
from doctr import transforms as T
from doctr.datasets import VOCABS
from doctr.models import recognition
from doctr.utils.metrics import TextMatch

def get_wer(x,y):    
    return fastwer.score(x, y)
    
def get_cer(x,y):
    return fastwer.score(x, y, char_level=True)


def get_test_results(predictions, language):
    df = pd.DataFrame(predictions)
    df[['pred','score']] =  pd.DataFrame(df.pred.tolist(), index= df.index)
    df = df.drop_duplicates()
    df['id']= df['name'].str.split("_")
    df[['temp','id']] =  pd.DataFrame(df.id.tolist(), index= df.index)
    df['id'] = df['id'].apply(lambda x: str(x).rstrip('.jpg'))
    df['id'] = df['id'].astype(int)
    df['name'] = df['name'].str.replace('_','/')
    df = df.sort_values('id')
    df = df[['name','pred']]
    filename = './data/results/'+language+'_results.txt'
    df.to_csv(filename, sep='\t', index=False)
    
def get_val_results(predictions, language):
    df = pd.DataFrame(predictions)
    df[['pred','score']] =  pd.DataFrame(df.pred.tolist(), index= df.index)
    df = df.drop_duplicates()
    df['id']= df['name'].str.split("_")
    if(language=='hindi' or language=='telugu' or language=='devanagari'):
        print("running "+language)
    else: 
        df[['temp','id']] =  pd.DataFrame(df.id.tolist(), index= df.index)
        df['id'] = df['id'].apply(lambda x: str(x).rstrip('.jpg'))
        df['id'] = df['id'].astype(int)
        df['name'] = df['name'].str.replace('_','/')
        df = df.sort_values('id')
    df = df[['name','pred', 'actual']]
    df['WER'] = df.apply(lambda x: get_wer([x['pred']], [x['actual']]), axis=1)
    df['CER'] = df.apply(lambda x: get_cer([x['pred']], [x['actual']]), axis=1)
    
    pred_list =  list(df["pred"])
    actual_list = list(df['actual'])

    WER = get_wer(pred_list, actual_list)
    CER = get_cer(pred_list, actual_list)
    print("\n Evaluation Results for "+language+" validation set: ")
    print("\tWord Error Rate = ",WER)
    print("\tChar Error Rate = ",CER)

    print("\tWord Recognition Rate = ", (100 - WER))
    print("\tChar Recognition Rate = ", (100 - CER))
    filename = './data/results/'+language+'_val_results.txt'
    df.to_csv(filename, sep='\t', index=False)

@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    predictions = []
    for images, targets, names in tqdm(val_loader):
        try:
            if torch.cuda.is_available():
                images = images.cuda()
            images = batch_transforms(images)
            if amp:
                with torch.cuda.amp.autocast():
                    out = model(images, targets, return_preds=True)
            else:
                out = model(images, targets, return_preds=True)
            # Compute metric
            
            d = {}
            d['pred'] = out['preds'][0]
            d['actual'] = targets[0]
            d['name'] = names[0]
            predictions.append(d)
            if len(out["preds"]):
                words, _ = zip(*out["preds"])
            else:
                words = []
            val_metric.update(targets, words)

            val_loss += out["loss"].item()
            batch_cnt += 1
        except ValueError:
            print(f"unexpected symbol/s in targets:\n{targets} \n--> skip batch")
            continue
            
    val_loss /= batch_cnt
    result = val_metric.summary()
    return val_loss, result["raw"], result["unicase"], predictions


def main(args):
    print(args)

    torch.backends.cudnn.benchmark = True

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    # Load doctr model
    model = recognition.__dict__[args.arch](
        pretrained=True if args.resume is None else False,
        input_shape=(3, args.input_size, 4 * args.input_size),
        vocab=VOCABS[args.vocab],
    ).eval()

    # Resume weights
    if isinstance(args.resume, str):
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)

    st = time.time()
    ds = datasets.__dict__[args.dataset](
        train=True,
        download=True,
        recognition_task=True,
        language=args.vocab,
        inp_path=args.input_path,
        sets=args.sets,
        use_polygons=args.regular,
        img_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
    )

#     _ds = datasets.__dict__[args.dataset](
#         train=False,
#         download=True,
#         recognition_task=True,
#         use_polygons=args.regular,
#         img_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
#     )
#     ds.data.extend([(np_img, target, name) for np_img, target, name in _ds.data])

    test_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(ds),
        pin_memory=torch.cuda.is_available(),
        collate_fn=ds.collate_fn,
    )
    print(f"Test set loaded in {time.time() - st:.4}s ({len(ds)} samples in " f"{len(test_loader)} batches)")

    mean, std = model.cfg["mean"], model.cfg["std"]
    batch_transforms = Normalize(mean=mean, std=std)

    # Metrics
    val_metric = TextMatch()

    # GPU
    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
    # Silent default switch to GPU if available
    elif torch.cuda.is_available():
        args.device = 0
    else:
        print("No accessible GPU, targe device set to CPU.")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()

    print("Running evaluation")
    val_loss, exact_match, partial_match, predictions = evaluate(model, test_loader, batch_transforms, val_metric, amp=args.amp)
#     print(f"Validation loss: {val_loss:.6} (Exact: {exact_match:.2%} | Partial: {partial_match:.2%})")
    if(args.sets != 'test'):
        get_val_results(predictions, args.vocab)
    else:
        get_test_results(predictions, args.vocab)
        


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR evaluation script for text recognition (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("arch", type=str, help="text-recognition model to evaluate")
    parser.add_argument("--vocab", type=str, default="hindi", help="Vocab to be used for evaluation")
    parser.add_argument("--dataset", type=str, default="IndicData", help="Dataset to evaluate on")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--input_size", type=int, default=32, help="input size H for the model, W = 4*H")
    parser.add_argument("--input_path", type=str, default="./data/processed/", help="Path of the test dataset")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--sets", type=str, default='test', help='Evaluating on Test or Validation set')
    parser.add_argument(
        "--only_regular", dest="regular", action="store_true", help="test set contains only regular text"
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
