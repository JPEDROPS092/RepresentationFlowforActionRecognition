# 🚀 Como Usar com seu Venv Existente

## Passo a Passo Rápido

### 1. Ative seu Venv
```bash
# Use o comando que você normalmente usa, por exemplo:
source venv/bin/activate
# ou
source .venv/bin/activate  
# ou
conda activate seu_env
```

### 2. Instale Dependências Necessárias
```bash
# Instalar pacotes principais
pip install torch torchvision opencv-python scikit-learn
pip install matplotlib seaborn tqdm pandas PyYAML tensorboard
pip install kaggle  # Para baixar dataset
```

### 3. Teste as Importações
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torchvision; print('TorchVision:', torchvision.__version__)"
```

### 4. Configure o Dataset Jester
```bash
python real_datasets_only.py --dataset jester
```

### 5. Baixe o Dataset via Kaggle

#### Configurar Kaggle API:
```bash
# 1. Vá em kaggle.com/account e baixe kaggle.json
# 2. Configure:
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Baixar dataset:
```bash
cd real_datasets/jester
kaggle datasets download -d toxicmender/20bn-jester
unzip 20bn-jester.zip
```

### 6. Execute o Treinamento
```bash
python complete_training_pipeline.py --config real_datasets/jester_config.yaml
```

## 🔧 Ou Use o Script Automático

```bash
# Executa tudo automaticamente
./run_with_jester.sh
```

## 📋 Verificar se Está Funcionando

### Teste Importações:
```bash
python test_imports.py
```

### Teste Dataset (mesmo sem vídeos):
```bash
python jester_dataset_loader.py
```

### Teste Modelo:
```bash
python -c "
from gesture_recognition_model import create_gesture_model
import torch
model = create_gesture_model('lightweight', num_classes=27)
x = torch.randn(1, 3, 8, 64, 64)
y = model(x)
print('Modelo OK, shape:', y.shape)
"
```

## 🎯 Estrutura Esperada Após Setup

```
RepresentationFlowforActionRecognition/
├── real_datasets/
│   └── jester/
│       ├── videos/           # 148K vídeos .mp4
│       ├── jester-v1-train.csv
│       ├── jester-v1-validation.csv
│       ├── jester-v1-labels.csv
│       └── jester_config.yaml
├── checkpoints_jester/       # Será criado durante treinamento
├── logs_jester/              # Logs do treinamento
└── evaluation_jester/        # Resultados da avaliação
```

## ⚡ Comandos Rápidos

```bash
# Ativar venv + treinar
source venv/bin/activate && python complete_training_pipeline.py --config real_datasets/jester_config.yaml

# Só avaliação (com modelo treinado)  
source venv/bin/activate && python complete_training_pipeline.py --config real_datasets/jester_config.yaml --eval-only --checkpoint checkpoints_jester/best_checkpoint.pth

# Teste rápido (sem download do dataset)
source venv/bin/activate && python test_imports.py
```

## 🔍 Solução de Problemas

### Erro: "No module named torch"
```bash
# Verificar se venv está ativo
echo $VIRTUAL_ENV  # Deve mostrar caminho do venv
pip install torch torchvision
```

### Erro: "Dataset not found" 
```bash
# Verificar se dataset foi baixado
ls real_datasets/jester/videos/ | wc -l  # Deve mostrar ~148000
```

### Erro: "CUDA out of memory"
```bash
# Editar jester_config.yaml:
# batch_size: 4  # Reduzir de 16 para 4
```

### Erro: "Kaggle not found"
```bash
pip install kaggle
# Configure credenciais como mostrado acima
```

## 🎛️ Configurações do Jester

O arquivo `jester_config.yaml` está otimizado para:
- **27 classes** de gestos
- **Modelo completo** com Representation Flow
- **Batch size 16** (reduza para 4-8 se pouca memória)
- **16 frames** por clipe
- **30 épocas** de treinamento

### Para treinar mais rápido (teste):
```yaml
training:
  batch_size: 4
  num_epochs: 5
  
data_preprocessing:
  clip_len: 8
  crop_size: 64
```

### Para máxima precisão:
```yaml
model:
  type: full
  flow_channels: 128
  
training:
  batch_size: 32  # Se tiver GPU com muita memória
  num_epochs: 50
```

## 📊 Resultados Esperados

- **Tempo de treinamento**: 2-6 horas (depende da GPU)
- **Acurácia esperada**: 85-95% no dataset Jester
- **Uso de memória**: ~4-8GB GPU (batch_size=16)
- **Tamanho do modelo**: ~50MB (checkpoint completo)

---

**🚀 Para começar agora:**
```bash
source seu_venv/bin/activate
./run_with_jester.sh
```