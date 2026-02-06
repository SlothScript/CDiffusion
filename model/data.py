import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import random

class MLMDataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer_path="tokenizer.json",
        max_seq_len=512,
        mask_rate_min=0.15,
        mask_rate_max=0.85,
        ignore_index=-100,
        max_data=0,
    ):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        if max_data == 0:
            self.texts = texts
        else:
            self.texts = texts[0:max_data]
            print(self.texts)
        self.max_seq_len = max_seq_len
        self.mask_rate_min = mask_rate_min
        self.mask_rate_max = mask_rate_max
        self.ignore_index = ignore_index

        self.pad_id  = self.tokenizer.token_to_id("[PAD]")
        self.mask_id = self.tokenizer.token_to_id("[MASK]")

        self.special_ids = {
            i for i in [self.pad_id, self.mask_id]
            if i is not None
        }

        self.vocab_size = self.tokenizer.get_vocab_size()

    def __len__(self):
        return len(self.texts)

    def _encode(self, text):
        ids = self.tokenizer.encode(text).ids[: self.max_seq_len]
        if len(ids) < self.max_seq_len:
            ids += [self.pad_id] * (self.max_seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def _mask_tokens(self, input_ids):
        labels = input_ids.clone()

        mlm_prob = random.uniform(self.mask_rate_min, self.mask_rate_max)
        prob = torch.full(labels.shape, mlm_prob)

        for sid in self.special_ids:
            prob[input_ids == sid] = 0.0

        masked = torch.bernoulli(prob).bool()
        labels[~masked] = self.ignore_index

        replace = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
        input_ids[replace] = self.mask_id

        rand = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked
            & ~replace
        )
        random_tokens = torch.randint(0, self.vocab_size, labels.shape)
        input_ids[rand] = random_tokens[rand]

        return input_ids, labels

    def __getitem__(self, idx):
        input_ids = self._encode(self.texts[idx])
        input_ids, labels = self._mask_tokens(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels
        }