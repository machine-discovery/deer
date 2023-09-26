"""
Code adapted from
https://github.com/RuslanKhalitov/ChordMixer/tree/main/dataloaders
"""

from pathlib import Path

import pickle
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F

import torchtext
from datasets import load_dataset, DatasetDict, Value


class AANDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_workers: int = 1,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.max_length = 4000
        self.append_bos = False
        self.append_eos = True
        self.l_max = 4000
        self.n_workers = num_workers
        self.cache_dir = self.get_cache_dir()

    def prepare_data(self):
        self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return

        torch.multiprocessing.set_sharing_strategy("file_system")

        dataset, self.tokenizer, self.vocab = self.process_dataset()
        print("AAN vocab size:", len(self.vocab))

        dataset.set_format(
            type="torch",
            columns=[
                "input_ids1",
                "input_ids2",
                "label"])
        self._train_dataset, self._val_dataset, self._test_dataset = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs1, xs2, ys = zip(
                *[
                    (data["input_ids1"], data["input_ids2"], data["label"])
                    for data in batch
                ]
            )
            xs1 = nn.utils.rnn.pad_sequence(
                xs1, padding_value=self.vocab["<pad>"], batch_first=True
            )
            xs2 = nn.utils.rnn.pad_sequence(
                xs2, padding_value=self.vocab["<pad>"], batch_first=True
            )
            L = max(xs1.size(1), xs2.size(1))
            xs1 = F.pad(xs1, (0, L - xs1.size(1)), value=self.vocab["<pad>"])  # (nbatch, nseq)
            xs2 = F.pad(xs2, (0, L - xs2.size(1)), value=self.vocab["<pad>"])  # (nbatch, nseq)
            xs1 = xs1[..., None].type(torch.float32)  # (nbatch, nseq, ndim)
            xs2 = xs2[..., None].type(torch.float32)  # (nbatch, nseq, ndim)
            xs = torch.cat((xs1, xs2), dim=1)  # (nbatch, nseq * 2, ndim)
            ys = torch.tensor(ys)  # (nbatch)
            return xs, ys

        self._collate_fn = collate_batch

    def process_dataset(self):
        if self.cache_dir is not None:
            return self._load_from_cache()

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "new_aan_pairs.train.tsv"),
                "val": str(self.data_dir / "new_aan_pairs.eval.tsv"),
                "test": str(self.data_dir / "new_aan_pairs.test.tsv"),
            },
            delimiter="\t",
            column_names=["label", "input1_id", "input2_id", "text1", "text2"],
            keep_in_memory=True,
        )

        dataset = dataset.remove_columns(["input1_id", "input2_id"])
        new_features = dataset["train"].features.copy()
        new_features["label"] = Value("int32")
        dataset = dataset.cast(new_features)

        tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)

        def tokenize(example): return {
            "tokens1": tokenizer(example["text1"])[:l_max],
            "tokens2": tokenizer(example["text2"])[:l_max],
        }
        dataset = dataset.map(
            tokenize,
            remove_columns=["text1", "text2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens1"] + dataset["train"]["tokens2"],
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.append_bos else [])
                + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        def encode(text): return vocab(
            (["<bos>"] if self.append_bos else [])
            + text
            + (["<eos>"] if self.append_eos else [])
        )

        def numericalize(example): return {
            "input_ids1": encode(example["tokens1"]),
            "input_ids2": encode(example["tokens2"]),
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens1", "tokens2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        self._save_to_cache(dataset, tokenizer, vocab)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab):
        cache_dir = self.data_dir / self._cache_dir_name
        os.makedirs(str(cache_dir), exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self):
        assert self.cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(self.cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(self.cache_dir))
        with open(self.cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(self.cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f"aan_{self.max_length}_{self.append_bos}_{self.append_eos}"

    def get_cache_dir(self):
        cache_dir = self.data_dir / self._cache_dir_name
        if cache_dir.is_dir():
            return cache_dir
        else:
            return None

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self._collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
        return test_dataloader


if __name__ == "__main__":
    import pdb
    dm = AANDataModule(
        data_dir="/home/yhl48/seq2seq/lra_release/lra_release/tsv_data"
    )
    dm.prepare_data()
    dm.setup()
    for i, batch in enumerate(dm.test_dataloader()):
        x1, x2, y = batch
        print(x1.shape, x2.shape, y.shape)
        pdb.set_trace()
