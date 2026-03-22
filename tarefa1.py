from datasets import load_dataset

def get_mini_dataset():
    # usando o multi30k (en/de) sugerido no lab
    dataset = load_dataset("bentrevett/multi30k")
    
    # pegando só 1000 amostras pra não travar tudo no treino
    train_subset = dataset['train'].select(range(1000))
    
    return train_subset

if __name__ == '__main__':
    ds = get_mini_dataset()
    print(f"Dataset carregado. Total de frases: {len(ds)}")