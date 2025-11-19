import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Callable, Optional
import math


# derived from IterableDataset instead of Dataset because of potentially large data size
# IterableDataset reads line by line from files steamingly, not loading all into memory.
class TranslationDataset(IterableDataset):
    """
    An IterableDataset for translation tasks. Reads from source and target
    files line by line, making it memory-efficient for large datasets.
    Can handle a directory of file pairs (e.g., train.0.en, train.0.zh, ...).
    """

    def __init__(self, data_path: str, src_suffix: str, tgt_suffix: str):
        """
        Args:
            data_path (str): Path to the data directory.
            src_suffix (str): Suffix for the source language files (e.g., 'en.ids').
            tgt_suffix (str): Suffix for the target language files (e.g., 'zh.ids').
        """
        super().__init__()
        p = Path(data_path)
        self.file_pairs = []

        if not p.is_dir():
            raise NotADirectoryError(
                f"Provided data_path is not a directory: {data_path}"
            )

        # Find all source files and create corresponding target file paths
        for src_file in sorted(p.glob(f"*.{src_suffix}")):
            # Assumes target file has the same name with a different suffix
            tgt_file = src_file.with_name(src_file.name.replace(src_suffix, tgt_suffix))
            if tgt_file.exists():
                self.file_pairs.append((str(src_file), str(tgt_file)))
            else:
                print(
                    f"Warning: Source file {src_file} found, but corresponding target file {tgt_file} is missing."
                )

        if not self.file_pairs:
            raise FileNotFoundError(
                f"No matching file pairs found in {data_path} with suffixes '{src_suffix}' and '{tgt_suffix}'"
            )

    def __iter__(self):
        """Yields a pair of source and target tensors for each line."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            iter_start = 0
            iter_end = len(self.file_pairs)
        else:  # multi-process data loading
            per_worker = int(
                math.ceil(len(self.file_pairs) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_pairs))

        for i in range(iter_start, iter_end):
            src_path, tgt_path = self.file_pairs[i]
            try:
                with open(src_path, "r", encoding="utf-8") as f_src, open(
                    tgt_path, "r", encoding="utf-8"
                ) as f_tgt:
                    for src_line, tgt_line in zip(f_src, f_tgt):
                        src_ids = list(map(int, src_line.split()))
                        tgt_ids = list(map(int, tgt_line.split()))
                        yield torch.tensor(src_ids), torch.tensor(tgt_ids)
            except Exception as e:
                print(f"Error reading file pair ({src_path}, {tgt_path}): {e}")
                continue


def create_dataloader(
    data_path: str,
    src_suffix: str,
    tgt_suffix: str,
    batch_size: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    num_workers: int = 0,
    worker_init_fn: Optional[Callable[[int], None]] = None,
):
    """
    Creates a DataLoader for the translation task using an IterableDataset.

    Args:
        data_path: Path to the directory containing tokenized files.
        src_suffix: Suffix for source language files.
        tgt_suffix: Suffix for target language files.
        batch_size: The number of samples per batch.
        pad_id, bos_id, eos_id: Special token IDs.
        num_workers: How many subprocesses to use for data loading. 0 means that
                     the data will be loaded in the main process. (default: 0)
        worker_init_fn: If not ``None``, this will be called on each worker
                        subprocess with the worker id as input. (default: None)

    Returns:
        A PyTorch DataLoader instance.
    """
    dataset = TranslationDataset(data_path, src_suffix, tgt_suffix)

    # Collate function to handle batching and padding for variable-length sequences
    def collate_fn(batch):
        """
        Processes a batch of sentence pairs to create padded tensors
        and the required tgt_in/tgt_out for teacher forcing.
        """
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)

        # src is for encoding, much simpler, just pad
        src_padded = pad_sequence(src_batch, batch_first=False, padding_value=pad_id)
        # tgt is for decoding, needs to create tgt_in and tgt_out for teacher forcing
        tgt_in_list = [torch.cat([torch.tensor([bos_id]), s]) for s in tgt_batch]
        tgt_out_list = [torch.cat([s, torch.tensor([eos_id])]) for s in tgt_batch]
        tgt_in_padded = pad_sequence(
            tgt_in_list, batch_first=False, padding_value=pad_id
        )
        tgt_out_padded = pad_sequence(
            tgt_out_list, batch_first=False, padding_value=pad_id
        )
        return src_padded, tgt_in_padded, tgt_out_padded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
