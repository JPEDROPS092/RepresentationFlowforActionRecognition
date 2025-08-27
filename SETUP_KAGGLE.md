# 🏆 Configurar Kaggle API - Guia Rápido

## 📋 Passo a Passo (2 minutos)

### 1. Criar conta no Kaggle (se não tiver)
- Vá em [kaggle.com](https://www.kaggle.com)
- Clique em "Register" e crie sua conta

### 2. Obter Token da API
1. Faça login em [kaggle.com](https://www.kaggle.com)
2. Clique no seu avatar (canto superior direito)
3. Selecione "Settings" ou "Account"
4. Na seção "API", clique em **"Create New API Token"**
5. Será baixado o arquivo `kaggle.json` automaticamente

### 3. Configurar no Sistema
```bash
# Criar pasta .kaggle
mkdir -p ~/.kaggle

# Copiar arquivo baixado (ajuste o caminho do Downloads)
cp ~/Downloads/kaggle.json ~/.kaggle/

# Definir permissões corretas
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Testar Configuração
```bash
# No seu venv ativado
pip install kaggle

# Testar
kaggle datasets list --max-size 1000
```

Se mostrar uma lista de datasets, está funcionando! ✅

## 🚀 Download Automático do Jester

Depois de configurar o Kaggle:

```bash
# Opção 1: Script automático completo
./run_with_jester.sh

# Opção 2: Só o download
python download_jester.py

# Opção 3: Manual
cd real_datasets/jester
kaggle datasets download -d toxicmender/20bn-jester
unzip 20bn-jester.zip
```

## 📊 Informações do Dataset Jester

- **Nome**: 20BN-Jester Dataset  
- **Tamanho**: ~22GB (148K vídeos)
- **Classes**: 27 gestos com as mãos
- **Kaggle**: [toxicmender/20bn-jester](https://www.kaggle.com/datasets/toxicmender/20bn-jester)

### Estrutura após download:
```
real_datasets/jester/
├── videos/                    # 148K vídeos .mp4
├── jester-v1-train.csv        # IDs de treinamento 
├── jester-v1-validation.csv   # IDs de validação
├── jester-v1-labels.csv       # Nome das classes
└── jester_config.yaml         # Configuração do modelo
```

## 🔧 Resolução de Problemas

### Erro: "kaggle command not found"
```bash
pip install kaggle
```

### Erro: "Unauthorized"
```bash
# Verificar se arquivo existe
ls -la ~/.kaggle/kaggle.json

# Reconfigurar permissões
chmod 600 ~/.kaggle/kaggle.json
```

### Erro: "Dataset not found"
- Verifique se o link está correto: `toxicmender/20bn-jester`
- Tente pelo navegador: [Kaggle Dataset](https://www.kaggle.com/datasets/toxicmender/20bn-jester)

### Erro de espaço em disco
- O dataset tem ~22GB
- Verifique espaço livre: `df -h`
- Se necessário, use um disco externo

## ⚡ Comandos Rápidos

```bash
# Setup completo em 1 comando (após configurar Kaggle)
source venv/bin/activate && ./run_with_jester.sh

# Só baixar dataset
source venv/bin/activate && python download_jester.py

# Verificar se dataset está completo
ls real_datasets/jester/videos/ | wc -l  # Deve mostrar ~148000
```

## 🎯 Próximos Passos

Após baixar o dataset:

1. **Verificar**: `python download_jester.py` (mostra se está completo)
2. **Treinar**: `python complete_training_pipeline.py --config real_datasets/jester_config.yaml`  
3. **Monitorar**: Os logs aparecem em `logs_jester/`
4. **Resultados**: Checkpoints salvos em `checkpoints_jester/`

---

**📱 Resumo:**
1. kaggle.com → Account → Create API Token
2. `cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
3. `./run_with_jester.sh` ✨