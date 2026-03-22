import torch
import torch.nn as nn
import torch.optim as optim

from tarefa1 import get_mini_dataset
from tarefa2 import tokenize_data
from meu_modelo import Transformer

# Adaptação da sua função do Lab 04 para PyTorch
def autoregressive_inference(model, encoder_input, tokenizer, max_new_tokens=32):
    START_IDX = tokenizer.cls_token_id
    EOS_IDX = tokenizer.sep_token_id

    model.eval() # Modo de inferência
    
    with torch.no_grad():
        # Passo 1: Codificar a entrada -> Z
        Z = model.encode(encoder_input)

        # Passo 2: Iniciar com token <START>
        decoder_input = torch.tensor([[START_IDX]])
        generated_tokens = []

        # Passo 3: Loop auto-regressivo
        step = 0
        while step < max_new_tokens:
            # Decoder prevê a distribuição sobre o vocabulário
            logits = model.decode(decoder_input, Z)

            # Greedy: token com maior probabilidade na ÚLTIMA posição
            next_idx = torch.argmax(logits[:, -1, :], dim=-1).item()

            # Critério de parada: token <EOS> (ou SEP, no caso do BERT)
            if next_idx == EOS_IDX:
                break

            generated_tokens.append(next_idx)

            # Concatenar novo token para a próxima iteração
            new_tok = torch.tensor([[next_idx]])
            decoder_input = torch.cat([decoder_input, new_tok], dim=1)

            step += 1

    # Retorna o texto decodificado de volta para humano
    return tokenizer.decode(generated_tokens)

# O Teste de Overfitting (Prova de Fogo)
def executar_prova_de_fogo():
    print(" Tarefa 4: A Prova de Fogo (Overfitting Test)")
    
    raw_data = get_mini_dataset()
    dataset_pronto, tokenizer = tokenize_data(raw_data)
    
    # 1. Pegando uma frase ESPECÍFICA do conjunto (a primeira)
    amostra = dataset_pronto[0]
    frase_original = raw_data[0]['en']
    frase_alvo = raw_data[0]['de']
    
    print(f"\n[ GABARITO ] Inglês: {frase_original}")
    print(f"[ GABARITO ] Alemão: {frase_alvo}\n")

    src = amostra['encoder_input_ids'].unsqueeze(0)
    tgt = amostra['decoder_input_ids'].unsqueeze(0)

    # 2. Instanciando o modelo
    model = Transformer(
        vocab_size=tokenizer.vocab_size, 
        d_model=128, num_heads=4, d_ff=256, num_layers=2, dropout=0.0 
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.002) 

    # 3. Forçando o Overfitting (Treinando a mesma frase várias vezes)
    epochs = 40
    print("Forçando a memorização da frase...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        tgt_input = tgt[:, :-1]
        tgt_esperado = tgt[:, 1:]
        
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_esperado.reshape(-1))
        
        loss.backward()
        optimizer.step()

    # 4. Chamando a sua função do Lab 04 para "vomitar" a tradução
    print("\nExecutando Loop Auto-regressivo...")
    traducao_gerada = autoregressive_inference(model, src, tokenizer)
    
    print("\n" + "="*60)
    print(f"TRADUÇÃO GERADA PELO MODELO: {traducao_gerada}")
    print("="*60)

if __name__ == '__main__':
    executar_prova_de_fogo()