import token
from transformers import AutoTokenizer
sentences="Hello world"
tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")
tokens_id=tokenizer(sentences).input_ids
print(tokens_id)
for token_id in tokens_id:
    print(tokenizer.decode(token_id))