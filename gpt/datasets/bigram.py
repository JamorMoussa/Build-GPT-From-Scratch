import torch 
from torch.utils.data import Dataset

__all__ = ["BigramDataset", "Tokenizer", ]


class Tokenizer:

    def __init__(self, vocab: list):
        self.vocab: list = vocab

        self.itos = {i: c for i, c in enumerate(self.vocab)}
        self.stoi = {c: i for i, c in enumerate(self.vocab)}

    def encode(self, string: str) -> torch.Tensor:
        return torch.tensor([self.stoi[s] for s in string]).type(torch.long)
    
    def decode(self, code: torch.Tensor) -> str:
        return "".join([self.itos[c] for c in code])


class BigramDatasetBase:

    def __init__(self, root: str, data: torch.Tensor = None, text: str = ""):

        self.root: str = root

        if data is None:
            self.text: str = self.open_text_file(self.root)
        else:
            self.data = data
            self.text = text

        self.process()

    def open_text_file(self, path: str):

        with open(path, "r") as f:
            text = f.read()

        return text 
    
    def get_vocab(self, text):
        return sorted(list(set(text)))
    
    def get_tokenizer(self) -> Tokenizer:
        return Tokenizer(self.vocab)
    
    def process(self,):

        self.vocab: list = self.get_vocab(self.text)
        self.n_vocab: int = len(self.vocab)

        self.data = self.get_tokenizer().encode(self.text)



class BigramDataset(BigramDatasetBase, Dataset):

    def __init__(self, root: str = "./", block_size: int = 8, data: torch.Tensor = None, text: str = "") -> None:
        super(BigramDataset, self).__init__(root= root, data = data, text= text)

        self.block_size: int = block_size

    def split_train_test(self, test_size: float = 0.1):

        n = int((1 - test_size) * len(self.data))

        return (
                BigramDataset(data=self.data[: n], text= self.text),     
                BigramDataset(data=self.data[n: ], text = self.text)
        )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        
        length = len(self.data) - self.block_size

        if idx > length: idx = length
        
        return self.data[idx: idx - self.block_size], self.data[idx + 1: idx - self.block_size + 1]



