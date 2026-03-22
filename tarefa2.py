from transformers import AutoTokenizer
from tarefa1 import get_mini_dataset

def tokenize_data(dataset, model_name="bert-base-multilingual-cased", max_len=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def process_batch(batch):
        # src (inglês) pro encoder, tgt (alemão) pro decoder
        src = tokenizer(
            batch['en'],
            padding='max_length',
            truncation=True,
            max_length=max_len
        )
        
        tgt = tokenizer(
            batch['de'],
            padding='max_length',
            truncation=True,
            max_length=max_len
        )
        
        return {
            'encoder_input_ids': src['input_ids'],
            'decoder_input_ids': tgt['input_ids']
        }

    # mapeando pro dataset todo
    tokenized_ds = dataset.map(process_batch, batched=True)
    
    # convertendo as colunas pra tensores do pytorch
    tokenized_ds.set_format(type='torch', columns=['encoder_input_ids', 'decoder_input_ids'])
    
    return tokenized_ds, tokenizer

if __name__ == '__main__':
    raw_data = get_mini_dataset()
    dados_prontos, tk = tokenize_data(raw_data)
    
    print("Shape encoder:", dados_prontos[0]['encoder_input_ids'].shape)
    print("Shape decoder:", dados_prontos[0]['decoder_input_ids'].shape)