import json
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel

warnings.filterwarnings("ignore")


class FrozenCLIPEmbedder(torch.nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        return outputs.last_hidden_state, outputs.pooler_output


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def clip_similarities(args):
    saving_points = 1
    total_num_texts = 0

    clipper = FrozenCLIPEmbedder()
    clipper.cuda()
    source_json = Path(args.db_path, 'source_list.json')
    source_inf = Path(args.db_path, 'source_list.inf')

    if source_json.exists():
        captions = open(source_json)
        captions = json.load(captions)
    else:
        with open(source_inf, 'r') as f:
            files_names = f.read().splitlines()

        captions = {}
        for fn in files_names:
            base_name = fn.split('.')[0]
            with open(Path(args.db_path, '{}.txt'.format(base_name)), 'r') as f:
                caption = f.readline()
                captions[fn] = caption

    parts = list(Path(args.ext_db_path).glob('*.parquet'))
    print('\n', captions)

    print('#Captions: {:,}, #Parquets: {}'.format(len(captions.keys()), len(parts)))
    cos = torch.nn.CosineSimilarity(dim=1)

    parts = np.sort(parts)
    # captions = np.sort(captions)

    similars = dict()

    for part in parts:
        data = pd.read_parquet(part, engine='pyarrow')
        texts = data['TEXT']
        num_text = len(texts)
        print('Parquet: {}, #Texts: {:,}'.format(part.name, num_text))

        for p_text in tqdm(chunks(list(texts), args.batch_size), total=num_text // args.batch_size):
            try:
                p_hidden_state, p_emb = clipper(p_text)
            except Exception as e:
                print(e)
                continue

            total_num_texts += len(p_text)
            for key in captions.keys():
                name = ''.join(key.split('.')[:-1])
                caption = captions[key]

                cap_hidden_state, cap_emb = clipper(caption)
                # Append self embeddings
                if name not in similars:
                    similars[name] = ([caption], [1.])

                scores = cos(p_emb, cap_emb).detach().cpu()
                ind = scores > args.threshold1
                ind = torch.nonzero(ind).squeeze(-1)
                if ind.shape[0] == 0:
                    continue

                similar = np.array(p_text)[ind]
                if type(similar) == np.str_:
                    similar = [similar]
                else:
                    similar = list(similar)

                similar_scores = list(scores[ind].cpu().numpy())
                similars[name] = (similars[name][0] + similar, similars[name][1] + similar_scores)

            if total_num_texts > saving_points * 1e6:
                str_obj = {}
                for key in similars.keys():
                    str_obj[key] = len(similars[key][0])

                print('\n', str_obj)

                print('saving similars...')
                file_name = '{}-th{}-total{}.pt'.format(args.dest_name1, args.threshold1, total_num_texts)
                torch.save(similars, Path(args.dest_folder1, file_name))
                saving_points += 1
                similars = {}

    file_name = '{}-th{}-total{}.pt'.format(args.dest_name1, args.threshold1, total_num_texts)
    torch.save(similars, Path(args.dest_folder1, file_name))
    print('Total Number of Texts: {:,}'.format(total_num_texts))


def filter_similars(args):
    similars = list(Path(args.dest_folder1).glob('*.pt'))
    similars = similars
    print('#Similars: {}'.format(len(similars)))
    filtered_similars = {}

    for similar in tqdm(similars):
        similar = torch.load(similar, map_location='cpu')
        for key in similar.keys():
            if key not in filtered_similars:
                filtered_similars[key] = [similar[key][0][0]]

            for i, s in enumerate(similar[key][1][1:]):
                if s > args.threshold2:
                    # starts [1:] => i + 1
                    filtered_similars[key] = filtered_similars[key] + [similar[key][0][i + 1]]

    for key in filtered_similars.keys():
        print(key, len(filtered_similars[key]))

    file_name = '{}-th{}-total{:}.pt'.format(args.dest_name2, args.threshold2, int(len(similars) * 1e6))
    torch.save(filtered_similars, Path(args.dest_folder2, file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--db_path",
        type=str,
        help="path of the dataset to load captions",
    )

    parser.add_argument(
        "-e",
        "--ext_db_path",
        type=str,
        help="path of the folders with parquet text files",
    )

    parser.add_argument(
        "-dn1",
        "--dest_name1",
        type=str,
        default="line-sim",
        help="destination filename prefix to save the similar similars",
    )

    parser.add_argument(
        "-df1",
        "--dest_folder1",
        type=str,
        help="destination folder to save the similar similars",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1500,
        help="batch size for texts",
    )

    parser.add_argument(
        "-th1",
        "--threshold1",
        type=float,
        default=0.5,
        help="thresholding the similarity score",
    )

    # ====================
    parser.add_argument(
        "-th2",
        "--threshold2",
        type=float,
        default=0.65,
        help="thresholding the similarity score for final filtering",
    )

    parser.add_argument(
        "-df2",
        "--dest_folder2",
        type=str,
        help="a folder for a single file containing all filtered similars",
    )

    parser.add_argument(
        "-dn2",
        "--dest_name2",
        default="line-sim-ready",
        type=str,
        help="a single file where filtered similars are stored for training",
    )

    args = parser.parse_args()
    print(args)

    clip_similarities(args)
    # filter_similars(args)



