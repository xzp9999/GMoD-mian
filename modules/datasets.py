import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.args = args
        self.top_np = self.__get_nounphrase()
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.img_labels, self.list_diseases = self.__get_labels(True)

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __get_nounphrase(self, top_k=100, file_name='count_nounphrase.json'):
        count_np = json.load(open(self.args.root_dir + file_name, 'r'))
        sorted_count_np = sorted([(k, v) for k, v in count_np.items()], key=lambda x: x[1], reverse=True)
        top_nounphrases = [k for k, v in sorted_count_np][:top_k]
        return top_nounphrases

    def __len__(self):
        return len(self.examples)

    def __get_labels(self, binary_mode):
        txt_file = self.args.root_dir + 'mimic-cxr-2.0.0-chexpert.csv'
        data = pd.read_csv(txt_file, dtype=object)
        label_names = list(data.columns.values[2:])

        data = data.to_numpy().astype(str)
        if binary_mode:
            data[data == '-1.0'] = "1"  # 2 Not sure
            data[data == 'nan'] = "0"  # 3 Not mentioned
        else:
            data[data == '-1.0'] = "2"  # 2 Not sure
            data[data == 'nan'] = "3"  # 3 Not mentioned

        img_labels = {}
        for i in range(len(data)):
            pid = 'p' + data[i, 0].item()
            sid = 's' + data[i, 1].item()
            labels = data[i, 2:].astype(float)
            img_labels[(pid, sid)] = labels
        return img_labels, label_names



class IuxrayMultiImageDataset(BaseDataset):

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        # 读label

        np_labels = np.zeros(len(self.top_np), dtype=float)
        for i in range(len(self.top_np)):
            if self.top_np[i] in example['report']:
                np_labels[i] = 1

        parts = example['image_path'][0].split('/')
        key_idx = (parts[1], parts[2])

        label = np.concatenate([self.img_labels[key_idx], np_labels])
        sample = (image_id, image, report_ids, report_masks, label, seq_length)

        return sample

