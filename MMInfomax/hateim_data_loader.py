from transformers import BertTokenizer

# path to a pretrained word embedding file
word_emb_path = '/home/diptesh/workspace/glove/glove.840B.300d.txt'

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# class HateMMDataset(Dataset):
#     def __init__(self, path):
        
