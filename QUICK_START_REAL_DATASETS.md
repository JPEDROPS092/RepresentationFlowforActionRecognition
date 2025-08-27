# Guia Rápido: Datasets Reais de Gestos

## 🎯 Opção Recomendada: Dataset Jester (20BN-Jester)

O **20BN-Jester** é o melhor dataset para começar:
- ✅ 27 classes de gestos com as mãos
- ✅ 148K vídeos de alta qualidade  
- ✅ Disponível no Kaggle (fácil download)
- ✅ Tamanho gerenciável (~22GB)
- ✅ Perfeito para Representation Flow

## 🚀 Setup Rápido (5 minutos)

### 1. Veja os Datasets Disponíveis
```bash
python real_datasets_only.py --dataset list
```

### 2. Setup do Jester
```bash
python real_datasets_only.py --dataset jester
```

### 3. Instalar Kaggle API
```bash
# No seu ambiente virtual
pip install kaggle
```

### 4. Configurar Credenciais Kaggle
1. Vá em [kaggle.com/account](https://www.kaggle.com/account) 
2. Clique em "Create New API Token"
3. Baixe o arquivo `kaggle.json`
4. Configure:
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 5. Baixar o Dataset
```bash
cd real_datasets/jester
kaggle datasets download -d toxicmender/20bn-jester
unzip 20bn-jester.zip
```

### 6. Treinar o Modelo
```bash
python complete_training_pipeline.py --config real_datasets/jester_config.yaml
```

## 📁 Estrutura Esperada do Jester

```
real_datasets/jester/
├── videos/
│   ├── 00001.mp4
│   ├── 00002.mp4
│   └── ... (148K vídeos)
├── jester-v1-train.csv       # IDs de treinamento
├── jester-v1-validation.csv  # IDs de validação  
├── jester-v1-labels.csv      # Classes dos gestos
└── jester_config.yaml        # Configuração gerada
```

## 🎬 Classes do Jester (27 gestos)

- **Swipes**: Left, Right, Up, Down
- **Push/Pull**: Hand Away/In, Two Fingers Away/In
- **Slides**: Two Fingers Left/Right/Up/Down
- **Rotations**: Hand Forward/Backward, Clockwise/Counter
- **Zoom**: Full Hand In/Out, Two Fingers In/Out
- **Gestos**: Thumb Up/Down, Shaking Hand, Stop Sign
- **Outros**: Drumming Fingers, No Gesture, Other

## 📊 Configuração do Jester

O arquivo `jester_config.yaml` é otimizado para o dataset:

```yaml
model:
  type: full              # Modelo completo
  num_classes: 27         # 27 gestos
  flow_channels: 64       # Canais de fluxo
  use_adaptive_flow: true # Fluxo adaptativo

training:
  batch_size: 16          # Batch adequado
  num_epochs: 30          # Épocas suficientes
  clip_len: 16            # 16 frames por clipe
  crop_size: 112          # Resolução espacial
```

## 🔄 Outros Datasets Reais

### ChaLearn IsoGD
```bash
python real_datasets_only.py --dataset chalearn
# Kaggle: kaggle datasets download -d shivamb/chalearn-isolated-sign-language-dataset
```

### EgoGesture  
```bash
# Requer registro manual em:
# http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html
```

### NVGesture
```bash
# Requer registro na NVIDIA:
# https://research.nvidia.com/publication/2016-06_online-detection-and
```

## ⚡ Teste Rápido

Para testar sem baixar o dataset completo:

```bash
# 1. Setup
python real_datasets_only.py --dataset jester

# 2. Teste o loader (mesmo sem vídeos)
python jester_dataset_loader.py

# 3. Teste o modelo  
python test_imports.py
```

## 🎯 Resultados Esperados

Com o dataset Jester:
- **Acurácia**: 85-95% (estado da arte ~97%)
- **Treinamento**: 2-4 horas (GPU)
- **Inference**: 45+ FPS (modelo completo)
- **Inference**: 120+ FPS (modelo leve)

## 🔧 Resolução de Problemas

### Dataset não encontrado
```bash
# Verifique se o download foi completo
ls real_datasets/jester/videos/ | wc -l  # Deve ter ~148K arquivos
```

### Erro de memória
```bash
# Reduza batch_size no config:
batch_size: 8  # ou 4
```

### Treinamento lento
```bash
# Use modelo lightweight:
model:
  type: lightweight
```

## 📈 Próximos Passos

1. **Treine com Jester**: Comece com o dataset mais acessível
2. **Experimente hiperparâmetros**: Ajuste batch_size, learning_rates
3. **Teste outros datasets**: ChaLearn para mais classes
4. **Deploy real-time**: Use o modelo lightweight para aplicações

---

**🚀 Comando para começar agora:**
```bash
python real_datasets_only.py --dataset jester
```