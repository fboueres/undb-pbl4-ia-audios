# Detecção de Sentimentos em Áudio com Inteligência Artificial

Projeto desenvolvido para o desafio de Processamento de Áudio e Inteligência Artificial utilizando classificação de sentimentos em fala humana.

O sistema utiliza processamento de sinais de áudio, extração de características acústicas e Redes Neurais Convolucionais (CNNs) para classificar trechos de fala em:

- negativo
- neutro
- positivo

---

# Objetivo

Desenvolver um sistema de Inteligência Artificial capaz de detectar automaticamente sentimentos em áudios de fala humana, simulando aplicações de atendimento ao cliente e monitoramento de experiência do consumidor.

---

# Tecnologias Utilizadas

- Python 3.11
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

# Dataset

O projeto utiliza o dataset:

CREMA-D — Crowd-Sourced Emotional Multimodal Actors Dataset

Fonte:
https://www.kaggle.com/datasets/ejlok1/cremad

As emoções originais do dataset foram convertidas para sentimentos:

| Emoção | Sentimento |
|---|---|
| HAP | positivo |
| NEU | neutro |
| ANG | negativo |
| SAD | negativo |
| FEA | negativo |
| DIS | negativo |

---

# Estrutura do Projeto

```txt
.
├── dataset
│   ├── AudioWAV
│   └── cremad.zip
├── mise.toml
├── models
│   └── sentiment_cnn.keras
├── outputs
│   ├── classes.npy
│   ├── confusion_matrix.png
│   ├── X.npy
│   └── y.npy
├── requirements.txt
└── src
    ├── build_features.py
    ├── config.py
    ├── download_dataset.py
    ├── evaluate.py
    ├── load_dataset.py
    ├── main.py
    ├── preprocess.py
    ├── predict.py
    ├── test_preprocess.py
    ├── train.py
    └── utils.py
```

---

# Instalação

## Clonar o repositório

```bash
git clone <repo>
cd projeto-audio-ai
```

---

## Configurar ambiente virtual

```bash
python -m venv .venv
```

### Linux / Mac

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

---

## Instalar dependências

```bash
pip install -r requirements.txt
```

---

# Configuração do Kaggle

Criar arquivo `.env` na raiz do projeto:

```env
KAGGLE_USERNAME=seu_usuario
KAGGLE_KEY=sua_chave
```

As credenciais podem ser obtidas em:

https://www.kaggle.com/settings

---

# Download do Dataset

```bash
python src/download_dataset.py
```

---

# Pipeline do Projeto

## 1. Verificar dataset

```bash
python src/check_dataset.py
```

---

## 2. Ler metadata

```bash
python src/load_dataset.py
```

---

## 3. Extrair MFCC e gerar features

```bash
python src/build_features.py
```

Esse processo gera:

- `outputs/X.npy`
- `outputs/y.npy`
- `outputs/classes.npy`

---

## 4. Treinar modelo

```bash
python src/train.py
```

O modelo treinado será salvo em:

```txt
models/sentiment_cnn.keras
```

---

## 5. Avaliar modelo

```bash
python src/evaluate.py
```

Resultados:

- métricas de classificação
- matriz de confusão
- `outputs/confusion_matrix.png`

---

## 6. Fazer predição

```bash
python src/predict.py caminho_do_audio.wav
```

Exemplo:

```bash
python src/predict.py dataset/AudioWAV/1001_DFA_HAP_XX.wav
```

Saída esperada:

```txt
Sentimento previsto: positivo
Confiança: 84.23%
```

---

# Arquitetura do Modelo

O modelo utiliza uma CNN composta por:

- Camadas Convolucionais
- MaxPooling
- Dropout
- Camadas Densas
- Softmax para classificação final

Entrada:
- MFCCs extraídos do áudio

Saída:
- negativo
- neutro
- positivo

---

# Pré-processamento

As seguintes etapas são aplicadas:

- Padronização de sample rate
- Normalização de áudio
- Extração de MFCC
- Padding de sequências
- Conversão para tensores

---

# Métricas Utilizadas

- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de Confusão

---

# Aplicação Prática

O sistema pode ser utilizado em:

- Call centers
- SAC
- Atendimento ao cliente
- Monitoramento de experiência
- Indicadores emocionais de chamadas

Exemplo:
- identificar clientes irritados;
- medir evolução emocional durante o atendimento;
- monitorar qualidade operacional.

---

# Limitações

- O modelo não interpreta texto ou significado semântico.
- O sistema analisa apenas padrões acústicos da voz.
- O dataset utiliza emoções simuladas por atores.

---

# Melhorias Futuras

- Mel-Spectrogramas
- Data Augmentation
- Batch Normalization
- APIs REST
- Predição em tempo real
- Dashboard de monitoramento
- Transfer Learning
- Transformers de áudio

---

# Licença

Projeto acadêmico desenvolvido para fins educacionais.
