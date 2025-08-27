#!/usr/bin/env python3
"""
Script para baixar automaticamente o dataset Jester via Kaggle API.
"""

import os
import subprocess
import sys
from pathlib import Path
import zipfile
import shutil


def check_kaggle_setup():
    """Verifica se a Kaggle API está configurada."""
    print("🔍 Verificando configuração do Kaggle...")
    
    # Verifica se kaggle está instalado
    try:
        import kaggle
        print("✓ Kaggle API instalada")
    except ImportError:
        print("❌ Kaggle API não instalada")
        print("Instalando...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        import kaggle
        print("✓ Kaggle API instalada")
    
    # Verifica credenciais
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_path.exists():
        print(f"✓ Credenciais encontradas: {kaggle_path}")
        return True
    else:
        print("❌ Credenciais do Kaggle não encontradas")
        print("\n📋 Para configurar:")
        print("1. Vá em https://www.kaggle.com/account")
        print("2. Clique em 'Create New API Token'")
        print("3. Baixe o arquivo kaggle.json")
        print("4. Execute:")
        print(f"   mkdir -p {Path.home() / '.kaggle'}")
        print(f"   cp kaggle.json {kaggle_path}")
        print(f"   chmod 600 {kaggle_path}")
        return False


def download_jester_dataset():
    """Baixa o dataset Jester do Kaggle."""
    
    dataset_dir = Path("real_datasets/jester")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Baixando dataset Jester para {dataset_dir}...")
    
    # Muda para diretório do dataset
    original_dir = Path.cwd()
    os.chdir(dataset_dir)
    
    try:
        # Baixa dataset via Kaggle API
        print("🚀 Executando: kaggle datasets download -d toxicmender/20bn-jester")
        result = subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "toxicmender/20bn-jester"
        ], check=True, capture_output=True, text=True)
        
        print("✓ Download concluído!")
        
        # Lista arquivos baixados
        print("\n📁 Arquivos baixados:")
        for file in Path(".").glob("*.zip"):
            print(f"  - {file.name} ({file.stat().st_size / (1024**3):.1f}GB)")
        
        # Descompacta arquivo
        zip_files = list(Path(".").glob("*.zip"))
        if zip_files:
            zip_file = zip_files[0]
            print(f"\n📦 Descompactando {zip_file.name}...")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            print("✓ Descompactação concluída!")
            
            # Remove arquivo zip para economizar espaço
            print("🗑️  Removendo arquivo zip...")
            zip_file.unlink()
            
            # Lista estrutura final
            print("\n📁 Estrutura do dataset:")
            for item in sorted(Path(".").iterdir()):
                if item.is_dir():
                    count = len(list(item.iterdir())) if item.is_dir() else 0
                    print(f"  📂 {item.name}/ ({count} itens)")
                else:
                    print(f"  📄 {item.name}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro no download: {e}")
        print("Saída do erro:", e.stderr)
        return False
    
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False
    
    finally:
        # Volta para diretório original
        os.chdir(original_dir)


def verify_dataset():
    """Verifica se o dataset foi baixado corretamente."""
    print("\n🔍 Verificando dataset...")
    
    dataset_dir = Path("real_datasets/jester")
    
    # Verifica arquivos necessários
    required_files = [
        "jester-v1-train.csv",
        "jester-v1-validation.csv", 
        "jester-v1-labels.csv",
        "videos"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = dataset_dir / file_name
        if file_path.exists():
            if file_name == "videos":
                video_count = len(list(file_path.glob("*.mp4")))
                print(f"✓ {file_name}/ ({video_count} vídeos)")
                if video_count == 0:
                    missing_files.append(f"{file_name} (vazia)")
            else:
                print(f"✓ {file_name}")
        else:
            missing_files.append(file_name)
            print(f"❌ {file_name}")
    
    if missing_files:
        print(f"\n⚠️  Arquivos faltando: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ Dataset completo!")
        return True


def main():
    """Função principal."""
    print("🎬 Download Automático do Dataset Jester")
    print("=" * 50)
    
    # Verifica configuração do Kaggle
    if not check_kaggle_setup():
        print("\n❌ Configure o Kaggle primeiro, depois execute novamente.")
        sys.exit(1)
    
    # Verifica se dataset já existe
    if Path("real_datasets/jester/videos").exists():
        video_count = len(list(Path("real_datasets/jester/videos").glob("*.mp4")))
        if video_count > 0:
            print(f"\n✓ Dataset já existe ({video_count} vídeos)")
            verify_dataset()
            print("\n🚀 Para treinar:")
            print("python complete_training_pipeline.py --config real_datasets/jester_config.yaml")
            return
    
    # Confirma download
    print(f"\n📥 O dataset Jester (~22GB) será baixado para real_datasets/jester/")
    response = input("Continuar? (y/N): ").lower().strip()
    
    if response != 'y':
        print("❌ Download cancelado")
        sys.exit(1)
    
    # Baixa dataset
    if download_jester_dataset():
        if verify_dataset():
            print("\n🎉 Dataset Jester baixado com sucesso!")
            print("\n🚀 Próximos passos:")
            print("python complete_training_pipeline.py --config real_datasets/jester_config.yaml")
        else:
            print("\n⚠️  Dataset baixado mas com problemas. Verifique manualmente.")
    else:
        print("\n❌ Falha no download. Tente novamente ou baixe manualmente.")
        print("\n📋 Download manual:")
        print("1. cd real_datasets/jester")
        print("2. kaggle datasets download -d toxicmender/20bn-jester")
        print("3. unzip 20bn-jester.zip")


if __name__ == "__main__":
    main()