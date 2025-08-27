#!/usr/bin/env python3
"""
Setup para datasets reais de reconhecimento de gestos.
Foca apenas em datasets que realmente existem e estão disponíveis.
"""

import os
import json
import requests
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request


class RealGestureDatasets:
    """
    Gerenciador para datasets reais de reconhecimento de gestos.
    """
    
    def __init__(self, base_dir: str = "./real_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def list_available_real_datasets(self) -> Dict[str, Dict]:
        """Lista datasets reais disponíveis com links de download reais."""
        
        return {
            "20bn-jester": {
                "name": "20BN-Jester Dataset",
                "description": "27 classes de gestos com as mãos, 148K vídeos",
                "size": "22.8 GB",
                "classes": 27,
                "samples": 148092,
                "download_url": "https://20bn.com/datasets/jester/v1",
                "paper": "https://arxiv.org/abs/1706.13022",
                "license": "Creative Commons Attribution 4.0",
                "registration_required": True,
                "direct_download": False,
                "kaggle_available": True,
                "kaggle_url": "https://www.kaggle.com/datasets/toxicmender/20bn-jester"
            },
            
            "ego-gesture": {
                "name": "EgoGesture Dataset", 
                "description": "83 classes de gestos em primeira pessoa, 24K vídeos",
                "size": "2.95 GB",
                "classes": 83,
                "samples": 24161,
                "download_url": "http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html",
                "paper": "https://arxiv.org/abs/1808.07522",
                "license": "Academic use only",
                "registration_required": True,
                "direct_download": False,
                "kaggle_available": False
            },
            
            "chalearn-isogd": {
                "name": "ChaLearn IsoGD",
                "description": "249 classes de gestos isolados, 47K vídeos",
                "size": "7.49 GB", 
                "classes": 249,
                "samples": 47933,
                "download_url": "http://chalearnlap.cvc.uab.es/dataset/21/description/",
                "paper": "https://arxiv.org/abs/1606.06496",
                "license": "Academic use only",
                "registration_required": True,
                "direct_download": False,
                "kaggle_available": True,
                "kaggle_url": "https://www.kaggle.com/datasets/shivamb/chalearn-isolated-sign-language-dataset"
            },
            
            "nvgesture": {
                "name": "NVGesture Dataset",
                "description": "25 classes de gestos, dados RGB e depth",
                "size": "~7 GB",
                "classes": 25,
                "samples": 1532,
                "download_url": "https://research.nvidia.com/publication/2016-06_online-detection-and",
                "paper": "https://arxiv.org/abs/1606.07247",
                "license": "NVIDIA Research License",
                "registration_required": True,
                "direct_download": False,
                "kaggle_available": False
            },
            
            "something-something-v2": {
                "name": "Something-Something V2",
                "description": "174 classes de ações temporais com objetos",
                "size": "~19.4 GB",
                "classes": 174, 
                "samples": 220847,
                "download_url": "https://developer.qualcomm.com/software/ai-datasets/something-something",
                "paper": "https://arxiv.org/abs/1706.04261",
                "license": "Academic use only",
                "registration_required": True,
                "direct_download": False,
                "kaggle_available": False
            }
        }
    
    def setup_jester_from_kaggle(self) -> Dict[str, str]:
        """
        Setup para o dataset Jester via Kaggle.
        Requer Kaggle API configurada.
        """
        jester_dir = self.base_dir / "jester"
        jester_dir.mkdir(parents=True, exist_ok=True)
        
        print("Setup do dataset 20BN-Jester via Kaggle...")
        print("Requisitos:")
        print("1. Instalar Kaggle API: pip install kaggle")
        print("2. Configurar credenciais: ~/.kaggle/kaggle.json")
        print("3. Executar: kaggle datasets download -d toxicmender/20bn-jester")
        
        # Instruções detalhadas
        instructions = """
Para baixar o dataset Jester:

1. Instale a API do Kaggle:
   pip install kaggle

2. Faça login no Kaggle e vá em Account -> API -> Create New API Token
   Isso baixará kaggle.json

3. Mova o arquivo para ~/.kaggle/kaggle.json:
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json

4. Execute o download:
   cd real_datasets/jester
   kaggle datasets download -d toxicmender/20bn-jester
   unzip 20bn-jester.zip

5. O dataset terá esta estrutura:
   jester/
   ├── videos/
   │   ├── 00001.mp4
   │   ├── 00002.mp4
   │   └── ...
   ├── jester-v1-labels.csv
   ├── jester-v1-train.csv
   └── jester-v1-validation.csv
"""
        
        with open(jester_dir / "download_instructions.txt", "w") as f:
            f.write(instructions)
        
        print(f"Instruções salvas em: {jester_dir}/download_instructions.txt")
        
        # Criar estrutura de configuração esperada
        return {
            "data_path": str(jester_dir / "videos"),
            "train_annotation": str(jester_dir / "jester-v1-train.csv"), 
            "val_annotation": str(jester_dir / "jester-v1-validation.csv"),
            "labels_file": str(jester_dir / "jester-v1-labels.csv"),
            "download_instructions": str(jester_dir / "download_instructions.txt")
        }
    
    def setup_chalearn_from_kaggle(self) -> Dict[str, str]:
        """Setup para ChaLearn IsoGD via Kaggle."""
        chalearn_dir = self.base_dir / "chalearn"
        chalearn_dir.mkdir(parents=True, exist_ok=True)
        
        instructions = """
Para baixar o dataset ChaLearn IsoGD:

1. Instale a API do Kaggle (se não tiver):
   pip install kaggle

2. Execute o download:
   cd real_datasets/chalearn  
   kaggle datasets download -d shivamb/chalearn-isolated-sign-language-dataset
   unzip chalearn-isolated-sign-language-dataset.zip

3. O dataset terá estrutura com vídeos organizados por classes.
"""
        
        with open(chalearn_dir / "download_instructions.txt", "w") as f:
            f.write(instructions)
        
        return {
            "data_path": str(chalearn_dir / "videos"),
            "download_instructions": str(chalearn_dir / "download_instructions.txt")
        }
    
    def create_jester_config(self, jester_paths: Dict[str, str]) -> str:
        """Cria configuração para o dataset Jester."""
        
        # Classes do Jester (27 gestos)
        jester_classes = [
            "Swiping Left", "Swiping Right", "Swiping Down", "Swiping Up",
            "Pushing Hand Away", "Pulling Hand In", "Sliding Two Fingers Left",
            "Sliding Two Fingers Right", "Sliding Two Fingers Down", 
            "Sliding Two Fingers Up", "Pushing Two Fingers Away", 
            "Pulling Two Fingers In", "Rolling Hand Forward",
            "Rolling Hand Backward", "Turning Hand Clockwise", 
            "Turning Hand Counterclockwise", "Zooming In With Full Hand",
            "Zooming Out With Full Hand", "Zooming In With Two Fingers",
            "Zooming Out With Two Fingers", "Thumb Up", "Thumb Down",
            "Shaking Hand", "Stop Sign", "Drumming Fingers", 
            "No gesture", "Doing other things"
        ]
        
        config = {
            'data': {
                'dataset_type': 'jester',
                'data_path': jester_paths['data_path'],
                'train_annotation': jester_paths['train_annotation'],
                'val_annotation': jester_paths['val_annotation'],
                'labels_file': jester_paths['labels_file'],
                'class_names': jester_classes
            },
            'model': {
                'type': 'full',  # Modelo completo para o Jester
                'num_classes': 27,
                'flow_channels': 64,
                'use_adaptive_flow': True,
                'freeze_backbone': True,
                'dropout_rate': 0.4
            },
            'training': {
                'batch_size': 16,
                'num_epochs': 30,
                'learning_rates': {
                    'flow_layers': 1e-3,
                    'attention_layers': 5e-4,
                    'classifier': 1e-3,
                    'backbone': 1e-5
                },
                'optimizer': 'adamw',
                'scheduler': 'cosine_annealing',
                'early_stopping_patience': 10,
                'save_frequency': 5
            },
            'data_preprocessing': {
                'clip_len': 16,
                'crop_size': 112,
                'temporal_stride': 1,
                'normalize': True,
                'num_workers': 4
            },
            'paths': {
                'checkpoint_dir': './checkpoints_jester',
                'log_dir': './logs_jester', 
                'evaluation_dir': './evaluation_jester'
            },
            'evaluation': {
                'metrics_to_compute': ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'],
                'save_predictions': True,
                'benchmark_inference': True,
                'realtime_evaluation': False
            }
        }
        
        config_path = self.base_dir / "jester_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        return str(config_path)
    
    def show_dataset_info(self):
        """Mostra informações detalhadas sobre datasets disponíveis."""
        datasets = self.list_available_real_datasets()
        
        print("=" * 80)
        print("DATASETS REAIS DE RECONHECIMENTO DE GESTOS DISPONÍVEIS")
        print("=" * 80)
        
        for key, info in datasets.items():
            print(f"\n📁 {info['name']} ({key})")
            print(f"   📋 Descrição: {info['description']}")
            print(f"   📊 Classes: {info['classes']}")
            print(f"   🎬 Amostras: {info['samples']:,}")
            print(f"   💾 Tamanho: {info['size']}")
            print(f"   🔗 URL: {info['download_url']}")
            if info['kaggle_available']:
                print(f"   🏆 Kaggle: {info.get('kaggle_url', 'Disponível')}")
            print(f"   📄 Paper: {info['paper']}")
            print(f"   ⚖️  Licença: {info['license']}")
            if info['registration_required']:
                print("   ⚠️  Requer registro")
    
    def setup_dataset(self, dataset_name: str) -> str:
        """Setup de um dataset específico."""
        
        if dataset_name == "jester" or dataset_name == "20bn-jester":
            paths = self.setup_jester_from_kaggle()
            config_path = self.create_jester_config(paths)
            
            print(f"\n✅ Setup do Jester concluído!")
            print(f"📁 Diretório: {paths['data_path']}")
            print(f"⚙️  Config: {config_path}")
            print(f"📋 Instruções: {paths['download_instructions']}")
            
            return config_path
            
        elif dataset_name == "chalearn":
            paths = self.setup_chalearn_from_kaggle()
            print(f"\n✅ Setup do ChaLearn iniciado!")
            print(f"📋 Instruções: {paths['download_instructions']}")
            return ""
            
        else:
            print(f"❌ Dataset '{dataset_name}' não suportado ainda.")
            print("Datasets disponíveis: jester, chalearn")
            return ""


def main():
    """Interface principal para setup de datasets reais."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup de datasets reais de gestos')
    parser.add_argument('--dataset', type=str, 
                       choices=['jester', '20bn-jester', 'chalearn', 'list'],
                       default='list',
                       help='Nome do dataset ou "list" para listar disponíveis')
    parser.add_argument('--base_dir', type=str, default='./real_datasets',
                       help='Diretório base para os datasets')
    
    args = parser.parse_args()
    
    manager = RealGestureDatasets(base_dir=args.base_dir)
    
    if args.dataset == 'list':
        manager.show_dataset_info()
        
        print("\n" + "=" * 80)
        print("RECOMENDAÇÕES:")
        print("=" * 80)
        print("🥇 20BN-Jester: Melhor para começar (disponível no Kaggle)")
        print("🥈 ChaLearn IsoGD: Muitas classes, bom para pesquisa")
        print("🥉 EgoGesture: Perspectiva única (primeira pessoa)")
        
        print("\n💡 Para começar com Jester:")
        print("   python real_datasets_only.py --dataset jester")
        
    else:
        config_path = manager.setup_dataset(args.dataset)
        
        if config_path:
            print("\n🚀 PRÓXIMOS PASSOS:")
            print("1. Baixe o dataset seguindo as instruções")
            print("2. Execute o treinamento:")
            print(f"   python complete_training_pipeline.py --config {config_path}")


if __name__ == "__main__":
    main()