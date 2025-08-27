import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from torchvision import transforms


class JesterDataset(Dataset):
    """
    Dataset loader para o 20BN-Jester Dataset.
    Formato específico do Jester com arquivos CSV.
    """
    
    def __init__(self,
                 data_path: str,
                 csv_file: str,
                 labels_file: str,
                 clip_len: int = 16,
                 crop_size: int = 112,
                 is_training: bool = True,
                 temporal_stride: int = 1):
        """
        Inicializa o dataset Jester.
        
        Args:
            data_path: Caminho para pasta com vídeos
            csv_file: Arquivo CSV com split (train/validation)
            labels_file: Arquivo CSV com labels das classes
            clip_len: Número de frames por clipe
            crop_size: Tamanho do crop espacial
            is_training: Se está em modo de treinamento
            temporal_stride: Stride temporal para amostragem
        """
        self.data_path = Path(data_path)
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.is_training = is_training
        self.temporal_stride = temporal_stride
        
        # Carrega labels das classes
        self.class_names = self._load_class_names(labels_file)
        self.num_classes = len(self.class_names)
        
        # Carrega split de dados
        self.samples = self._load_samples(csv_file)
        
        # Transformações
        self.transform = self._get_transforms()
        
        print(f"Dataset Jester carregado:")
        print(f"  Amostras: {len(self.samples)}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Modo: {'Treinamento' if is_training else 'Validação'}")
    
    def _load_class_names(self, labels_file: str) -> List[str]:
        """Carrega nomes das classes do arquivo de labels."""
        labels_path = Path(labels_file)
        
        if labels_path.exists():
            # Formato: class_id;class_name
            df = pd.read_csv(labels_path, sep=';', header=None, names=['class_id', 'class_name'])
            return df['class_name'].tolist()
        else:
            # Classes padrão do Jester se arquivo não existir
            return [
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
    
    def _load_samples(self, csv_file: str) -> List[Dict]:
        """Carrega amostras do arquivo CSV."""
        csv_path = Path(csv_file)
        samples = []
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_file}")
        
        # Formato: video_id;class_name ou video_id,class_name
        try:
            df = pd.read_csv(csv_path, sep=';', header=None, names=['video_id', 'class_name'])
        except:
            df = pd.read_csv(csv_path, sep=',', header=None, names=['video_id', 'class_name'])
        
        for _, row in df.iterrows():
            video_id = str(row['video_id']).zfill(5)  # Preenche com zeros: 00001
            class_name = row['class_name']
            
            # Verifica se vídeo existe
            video_path = self.data_path / f"{video_id}.mp4"
            if video_path.exists():
                samples.append({
                    'video_id': video_id,
                    'video_path': str(video_path),
                    'class_name': class_name,
                    'class_id': self.class_names.index(class_name) if class_name in self.class_names else 0
                })
            else:
                # Tenta outros formatos comuns
                for ext in ['.webm', '.avi', '.mov']:
                    video_path_alt = self.data_path / f"{video_id}{ext}"
                    if video_path_alt.exists():
                        samples.append({
                            'video_id': video_id,
                            'video_path': str(video_path_alt),
                            'class_name': class_name,
                            'class_id': self.class_names.index(class_name) if class_name in self.class_names else 0
                        })
                        break
        
        return samples
    
    def _get_transforms(self):
        """Obtém transformações de dados."""
        if self.is_training:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.RandomCrop(self.crop_size),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.crop_size, self.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """Carrega frames do vídeo."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Nenhum frame carregado de {video_path}")
        
        return np.array(frames)
    
    def _sample_frames(self, frames: np.ndarray) -> np.ndarray:
        """Amostra frames do vídeo."""
        total_frames = len(frames)
        
        if total_frames <= self.clip_len:
            # Repete frames se vídeo for muito curto
            indices = np.linspace(0, total_frames - 1, self.clip_len).astype(int)
        else:
            if self.is_training:
                # Amostragem aleatória para treinamento
                max_start = total_frames - (self.clip_len * self.temporal_stride)
                start_idx = random.randint(0, max(0, max_start))
                indices = np.arange(start_idx, 
                                  start_idx + self.clip_len * self.temporal_stride,
                                  self.temporal_stride)
            else:
                # Amostragem central para validação
                start_idx = (total_frames - self.clip_len * self.temporal_stride) // 2
                start_idx = max(0, start_idx)
                indices = np.arange(start_idx,
                                  start_idx + self.clip_len * self.temporal_stride,
                                  self.temporal_stride)
        
        # Garante que índices estão dentro dos limites
        indices = np.clip(indices, 0, total_frames - 1)
        return frames[indices]
    
    def _apply_transforms(self, frames: np.ndarray) -> torch.Tensor:
        """Aplica transformações espaciais aos frames."""
        transformed_frames = []
        
        for frame in frames:
            frame_tensor = self.transform(frame)
            transformed_frames.append(frame_tensor)
        
        # Empilha para criar (T, C, H, W) depois permuta para (C, T, H, W)
        clip_tensor = torch.stack(transformed_frames)  # (T, C, H, W)
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        return clip_tensor
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Obtém uma amostra do dataset."""
        try:
            sample = self.samples[idx]
            
            # Carrega frames do vídeo
            frames = self._load_video(sample['video_path'])
            
            # Amostra frames temporalmente
            sampled_frames = self._sample_frames(frames)
            
            # Aplica transformações espaciais
            clip_tensor = self._apply_transforms(sampled_frames)
            
            # Obtém label
            label = sample['class_id']
            
            return clip_tensor, label
            
        except Exception as e:
            print(f"Erro ao processar amostra {idx}: {e}")
            # Retorna amostra aleatória como fallback
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


def create_jester_dataloaders(
    data_path: str,
    train_csv: str,
    val_csv: str,
    labels_csv: str,
    batch_size: int = 16,
    clip_len: int = 16,
    crop_size: int = 112,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Cria dataloaders para o dataset Jester.
    
    Returns:
        train_loader, val_loader, class_names
    """
    
    # Dataset de treinamento
    train_dataset = JesterDataset(
        data_path=data_path,
        csv_file=train_csv,
        labels_file=labels_csv,
        clip_len=clip_len,
        crop_size=crop_size,
        is_training=True
    )
    
    # Dataset de validação
    val_dataset = JesterDataset(
        data_path=data_path,
        csv_file=val_csv,
        labels_file=labels_csv,
        clip_len=clip_len,
        crop_size=crop_size,
        is_training=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset.class_names


if __name__ == "__main__":
    # Teste do dataset
    print("Testando Jester Dataset Loader...")
    
    # Caminhos de exemplo (ajuste conforme sua estrutura)
    data_path = "./real_datasets/jester/videos"
    train_csv = "./real_datasets/jester/jester-v1-train.csv"
    val_csv = "./real_datasets/jester/jester-v1-validation.csv"
    labels_csv = "./real_datasets/jester/jester-v1-labels.csv"
    
    try:
        train_loader, val_loader, class_names = create_jester_dataloaders(
            data_path=data_path,
            train_csv=train_csv,
            val_csv=val_csv,
            labels_csv=labels_csv,
            batch_size=4,
            clip_len=8,
            crop_size=64
        )
        
        print(f"✅ Dataloaders criados com sucesso!")
        print(f"Classes: {len(class_names)}")
        print(f"Treinamento: {len(train_loader.dataset)} amostras")
        print(f"Validação: {len(val_loader.dataset)} amostras")
        
        # Teste uma amostra
        for videos, labels in train_loader:
            print(f"Forma do vídeo: {videos.shape}")
            print(f"Labels: {labels}")
            break
            
    except FileNotFoundError as e:
        print(f"❌ Arquivos não encontrados: {e}")
        print("Execute primeiro:")
        print("python real_datasets_only.py --dataset jester")
        
    except Exception as e:
        print(f"❌ Erro: {e}")