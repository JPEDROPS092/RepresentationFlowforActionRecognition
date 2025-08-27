#!/bin/bash

# Script para executar com venv existente e dataset Jester

echo "🚀 Executando Reconhecimento de Gestos com Dataset Jester"
echo "=========================================================="

# Ativar venv (ajuste o caminho conforme seu venv)
echo "📦 Ativando ambiente virtual..."

# Tente diferentes caminhos comuns de venv
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ venv ativado"
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
    echo "✓ ../venv ativado"
elif [ -d "env" ]; then
    source env/bin/activate
    echo "✓ env ativado"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ .venv ativado"
else
    echo "⚠️  Venv não encontrado automaticamente."
    echo "Por favor, ative manualmente seu venv primeiro:"
    echo "source seu_venv/bin/activate"
    echo ""
    echo "Depois execute:"
    echo "python complete_training_pipeline.py --config real_datasets/jester_config.yaml"
    exit 1
fi

echo ""
echo "🔍 Verificando ambiente..."
python --version
pip show torch | head -2 || echo "❌ PyTorch não instalado"

echo ""
echo "📊 Verificando configuração do Jester..."
if [ -f "real_datasets/jester_config.yaml" ]; then
    echo "✓ Configuração encontrada"
    echo "📁 Estrutura do dataset:"
    ls -la real_datasets/jester/ 2>/dev/null || echo "❌ Dataset não baixado ainda"
else
    echo "❌ Configuração não encontrada. Execute primeiro:"
    echo "python real_datasets_only.py --dataset jester"
    exit 1
fi

echo ""
echo "🎬 Status do dataset Jester:"
if [ -d "real_datasets/jester/videos" ]; then
    video_count=$(ls real_datasets/jester/videos/*.mp4 2>/dev/null | wc -l)
    if [ $video_count -gt 0 ]; then
        echo "✓ $video_count vídeos encontrados"
        echo ""
        echo "🚀 Iniciando treinamento..."
        python complete_training_pipeline.py --config real_datasets/jester_config.yaml
    else
        echo "❌ Pasta de vídeos vazia."
        echo ""
        echo "🚀 Iniciando download automático do dataset..."
        python download_jester.py
    fi
else
    echo "❌ Pasta de vídeos não existe."
    echo "🚀 Configurando e baixando dataset automaticamente..."
    python real_datasets_only.py --dataset jester
    python download_jester.py
fi