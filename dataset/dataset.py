from torch import tensor

from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        super(GPTDatasetV1,self).__init__()
        self.input_ids = []
        self.target_ids = []

        tokenized_text = tokenizer.encode(text)

        for i in range(0,len(tokenized_text)-max_length,stride):
            input_chunks = tokenized_text[i:i+max_length]
            target_chunk = tokenized_text[i+1:i+1+max_length]
            self.input_ids.append(tensor(input_chunks))
            self.target_ids.append(tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index],self.target_ids[index]
    
def create_dataloader(txt, batch_size=4, max_length=256,
    stride=128, shuffle=True, drop_last=True,
    num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt,tokenizer,max_length=max_length,stride=stride)

    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle,
        drop_last = drop_last,
        num_workers=num_workers
    )
    return dataloader