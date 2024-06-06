from gpt import models, datasets

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "Train a Bigram Language Model")

    parser.add_argument('--dataset', type= str, default="./",
                        help="")
    
    parser.add_argument('--lr', type= float, default= 0.01)
    
    parser.add_argument('--context_size', type= int, default= 8)

    parser.add_argument('--batch_size', type= int, default= 4)
    
    parser.add_argument('--test_size', type= float, default= 0.1)
        
    args = parser.parse_args()


    dataset = datasets.BigramDataset(
                    root= args.dataset,
                    block_size= args.context_size
            )
    
    train_set, test_set = dataset.split_train_test(test_size= args.test_size)

    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = models.BigramLanguageModel(dataset.n_vocab).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr= args.lr)


    for epoch in range(1):

        xb, yb  = next(iter(train_loader))

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(loss.item())

    
