import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importando os módulos que criamos nas Tarefas 1 e 2
from tarefa1 import get_mini_dataset
from tarefa2 import tokenize_data

# Importando o SEU modelo traduzido para PyTorch
from meu_modelo import Transformer 

def treinar_modelo():
    print("--- Tarefa 3: Motor de Otimização ---")
    
    # 1. Carregando e preparando os dados
    raw_data = get_mini_dataset()
    dataset_pronto, tokenizer = tokenize_data(raw_data)
    
    # DataLoader agrupa as frases em lotes (batches) de 16 para treinar mais rápido
    dataloader = DataLoader(dataset_pronto, batch_size=16, shuffle=True)
    
    # Pegando IDs importantes do tokenizador
    PAD_IDX = tokenizer.pad_token_id
    VOCAB_SIZE = tokenizer.vocab_size

    # 2. Instanciando o modelo
    # Usando dimensões menores (d_model=128) para não fritar a CPU e rodar rápido
    print("Instanciando o Transformer...")
    model = Transformer(
        vocab_size=VOCAB_SIZE, 
        d_model=128, 
        num_heads=4, 
        d_ff=256, 
        num_layers=2, 
        dropout=0.1
    )
    
    # 3. Definindo Função de Perda e Otimizador
    # ignore_index=PAD_IDX garante que o modelo não seja penalizado por errar os zeros do padding
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # 4. O Laço de Treinamento (Training Loop)
    epochs = 6
    print("\nIniciando o treinamento...")
    
    for epoch in range(epochs):
        model.train() # Coloca o modelo em modo de treino
        epoch_loss = 0
        
        for batch in dataloader:
            src = batch['encoder_input_ids'] 
            tgt = batch['decoder_input_ids'] 
            
            # --- TEACHER FORCING ---
            # O decoder recebe a frase inteira, exceto o último token
            tgt_input = tgt[:, :-1]
            # O gabarito (target) é a frase deslocada 1 posição para a direita
            tgt_esperado = tgt[:, 1:]
            
            # Zerando os gradientes da iteração anterior
            optimizer.zero_grad()
            
            # Forward: Passa os dados pelo Transformer
            logits = model(src, tgt_input) 
            
            # A CrossEntropyLoss exige que os tensores sejam achatados
            # logits vai de (batch_size, seq_len, vocab_size) para (batch_size * seq_len, vocab_size)
            logits_reshaped = logits.reshape(-1, logits.shape[-1])
            tgt_esperado_reshaped = tgt_esperado.reshape(-1)
            
            # Calcula o erro
            loss = criterion(logits_reshaped, tgt_esperado_reshaped)
            
            # Backward: Calcula as derivadas (aqui a mágica do PyTorch acontece)
            loss.backward()
            
            # Step: Atualiza os pesos (matrizes Wq, Wk, Wv, etc.)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Exibe a média do erro na época. O esperado é que esse valor caia a cada linha!
        media_loss = epoch_loss / len(dataloader)
        print(f"Época {epoch+1:02d}/{epochs} | Loss: {media_loss:.4f}")

if __name__ == '__main__':
    treinar_modelo()