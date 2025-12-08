import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import BrainMRIDataset, NiiDataAugmentation
from model import EfficientNet3D
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import logging
from datetime import datetime
import json

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )


def log_epoch_results(epoch, num_epochs, train_metrics, val_metrics):
    logging.info(f'\nEpoch [{epoch+1}/{num_epochs}]')

    logging.info(f'Train - Loss: {train_metrics["loss"]:.4f}')
    logging.info(f'Train - Accuracy: {train_metrics["accuracy"]:.4f}')
    logging.info(f'Train - Macro F1: {train_metrics["macro_f1"]:.4f}')
    logging.info(f'Train - Per Class F1: {train_metrics["per_class_f1"]}')

    logging.info(f'Val - Loss: {val_metrics["loss"]:.4f}')
    logging.info(f'Val - Accuracy: {val_metrics["accuracy"]:.4f}')
    logging.info(f'Val - Macro F1: {val_metrics["macro_f1"]:.4f}')
    logging.info(f'Val - Per Class F1: {val_metrics["per_class_f1"]}')
    
    if "class_distribution" in train_metrics:
        logging.info("\nClass Distribution:")
        class_names = ['Normal', 'Osteoporosis']
        for i, count in enumerate(train_metrics["class_distribution"]):
            percentage = (count / sum(train_metrics["class_distribution"])) * 100
            logging.info(f"{class_names[i]}: {count} ({percentage:.2f}%)")
            

def calculate_batch_metrics(y_true, y_pred, loss, n_batches):
    return {
        'loss': loss / n_batches,
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'per_class_f1': f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'class_distribution': np.bincount(y_true, minlength=2).tolist()
    }

def preload_data(train_loader_t1, train_loader_t2, val_loader_t1, val_loader_t2):
    logging.info("Preloading datasets to memory...")
    
    preloaded_data = {
        'train_t1': [],
        'train_t2': [],
        'train_labels': [],
        'val_t1': [],
        'val_t2': [],
        'val_labels': []
    }
    
    logging.info("Preloading training data...")
    for t1_batch, t2_batch in zip(train_loader_t1, train_loader_t2):
        t1_data, t1_labels = t1_batch
        t2_data, _ = t2_batch
        if t1_data.size(0) == t2_data.size(0):
            try:
                preloaded_data['train_t1'].append(t1_data.cuda(non_blocking=True))
                preloaded_data['train_t2'].append(t2_data.cuda(non_blocking=True))
                preloaded_data['train_labels'].append(t1_labels.cuda(non_blocking=True))
            except Exception as e:
                logging.error(f"Error preloading training batch: {str(e)}")
                continue
    
    logging.info("Preloading validation data...")
    for t1_batch, t2_batch in zip(val_loader_t1, val_loader_t2):
        t1_data, t1_labels = t1_batch
        t2_data, _ = t2_batch
        if t1_data.size(0) == t2_data.size(0):
            try:
                preloaded_data['val_t1'].append(t1_data.cuda(non_blocking=True))
                preloaded_data['val_t2'].append(t2_data.cuda(non_blocking=True))
                preloaded_data['val_labels'].append(t1_labels.cuda(non_blocking=True))
            except Exception as e:
                logging.error(f"Error preloading validation batch: {str(e)}")
                continue
    
    logging.info(f"Preloaded training batches: {len(preloaded_data['train_t1'])}")
    logging.info(f"Preloaded validation batches: {len(preloaded_data['val_t1'])}")
    
    return preloaded_data

def train_model():
    current_time = "2025"
    current_user = "qwqwqweq"
    setup_logging()
    logging.info(f"Starting training process... Using device: {device}")
    logging.info(f"Current time: {current_time}")
    logging.info(f"Current user: {current_user}")
    
    transform = NiiDataAugmentation(p=0.3)
    base_path = '/data/Dataset_Bone/_train/mri'
    batch_size = 16
    num_workers = 4
    num_epochs = 100
    
    train_dataset_t1 = BrainMRIDataset(base_path, 'T1', transform=transform, mode='train')
    train_dataset_t2 = BrainMRIDataset(base_path, 'T2', transform=transform, mode='train')
    
    val_dataset_t1 = BrainMRIDataset(base_path, 'T1', transform=None, mode='val')
    val_dataset_t2 = BrainMRIDataset(base_path, 'T2', transform=None, mode='val')
    
    validate_datasets([train_dataset_t1, train_dataset_t2, val_dataset_t1, val_dataset_t2])
    
    train_loader_t1 = DataLoader(train_dataset_t1, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True, drop_last=True)
    train_loader_t2 = DataLoader(train_dataset_t2, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader_t1 = DataLoader(val_dataset_t1, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader_t2 = DataLoader(val_dataset_t2, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    
    preloaded_data = preload_data(train_loader_t1, train_loader_t2, 
                                 val_loader_t1, val_loader_t2)
    
    model = EfficientNet3D(num_classes=3)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    class_weights = compute_class_weights(train_dataset_t1) 
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.05
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001, #0.005
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=3,
        T_mult=2,
        eta_min=1e-6
    )
    
    scaler = torch.amp.GradScaler()
    
    best_val_metrics = {
        'loss': float('inf'),
        'accuracy': 0,
        'macro_f1': 0,
        'weighted_f1': 0  
    }
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch(model, preloaded_data, criterion, 
                                  optimizer, scaler, epoch, num_epochs)
        val_metrics = validate_epoch(model, preloaded_data, criterion)
        scheduler.step()

        log_epoch_results(epoch, num_epochs, train_metrics, val_metrics)
        
        current_score = (val_metrics['macro_f1'] + val_metrics['weighted_f1']) / 2
        best_score = (best_val_metrics['macro_f1'] + best_val_metrics['weighted_f1']) / 2
        
        if current_score > best_score:
            best_val_metrics = val_metrics.copy()
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                timestamp=current_time,
                user=current_user
            )
    
    return model, best_val_metrics

def compute_class_weights(dataset):
    labels = [sample[1] for sample in dataset.samples]
    class_counts = np.bincount(labels)
   
    if len(class_counts) != 3:
        raise ValueError(f"Expected 3 classes, got {len(class_counts)}")
    total_samples = len(labels)
    weights = torch.FloatTensor(total_samples / (3 * class_counts))
    
    logging.info("Class distribution:")
    for i, (count, weight) in enumerate(zip(class_counts, weights)):
        class_name = ['Normal', 'Osteopenia', 'Osteoporosis'][i]
        logging.info(f"{class_name}: {count} samples, weight: {weight:.4f}")
    
    logging.info(f"Final class weights: {weights}")
    return weights

def validate_datasets(datasets):
    expected_labels = {0, 1, 2}
    
    for dataset in datasets:
        labels = [label for _, label in dataset.samples]
        unique_labels = set(np.unique(labels))
        
        if not unique_labels.issubset(expected_labels):
            raise ValueError(
                f"Dataset contains invalid labels: {unique_labels}\n"
                f"Expected labels should be in {expected_labels}"
            )
        
        if len(unique_labels) != 3:
            logging.warning(
                f"Dataset is missing some classes. Found classes: {unique_labels}\n"
                f"Expected all classes: {expected_labels}"
            )
        
        class_counts = np.bincount(labels, minlength=3)
        logging.info(f"Dataset class distribution:")
        class_names = ['Normal', 'Osteopenia', 'Osteoporosis']
        for i, count in enumerate(class_counts):
            logging.info(f"{class_names[i]}: {count} samples")
    
    logging.info("All datasets validated successfully")

def train_epoch(model, preloaded_data, criterion, optimizer, scaler, epoch, num_epochs):
    model.train()
    train_loss = 0
    train_pred = []
    train_true = []
    
    for batch_idx in range(len(preloaded_data['train_t1'])):
        try:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(
                    preloaded_data['train_t1'][batch_idx],
                    preloaded_data['train_t2'][batch_idx]
                )
                loss = criterion(outputs, preloaded_data['train_labels'][batch_idx])
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_pred.extend(predicted.cpu().numpy())
            train_true.extend(preloaded_data['train_labels'][batch_idx].cpu().numpy())
            
            if batch_idx % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}] "
                           f"Batch [{batch_idx}/{len(preloaded_data['train_t1'])}] "
                           f"Loss: {loss.item():.4f}")
                
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    return calculate_metrics(train_true, train_pred, train_loss, len(preloaded_data['train_t1']))

def validate_epoch(model, preloaded_data, criterion):
    model.eval()
    val_loss = 0
    val_pred = []
    val_true = []
    
    with torch.no_grad():
        for batch_idx in range(len(preloaded_data['val_t1'])):
            try:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(
                        preloaded_data['val_t1'][batch_idx],
                        preloaded_data['val_t2'][batch_idx]
                    )
                    loss = criterion(outputs, preloaded_data['val_labels'][batch_idx])
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_pred.extend(predicted.cpu().numpy())
                val_true.extend(preloaded_data['val_labels'][batch_idx].cpu().numpy())
                
            except Exception as e:
                logging.error(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    return calculate_metrics(val_true, val_pred, val_loss, len(preloaded_data['val_t1']))

def calculate_metrics(y_true, y_pred, loss_sum, n_batches):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    class_names = ['Normal', 'Osteopenia', 'Osteoporosis']
    per_class_precision = precision_score(y_true, y_pred, average=None).tolist()
    per_class_recall = recall_score(y_true, y_pred, average=None).tolist()
    per_class_f1 = f1_score(y_true, y_pred, average=None).tolist()
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'precision': per_class_precision[i],
            'recall': per_class_recall[i],
            'f1': per_class_f1[i]
        }
    
    metrics = {
        'loss': loss_sum / n_batches,
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'per_class_f1': per_class_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'confusion_matrix': conf_matrix.tolist(),
        'class_metrics': class_metrics,
        'timestamp': "2025",
        'user': "qwqwqweq"
    }
    
    logging.info(f"Metrics calculated at {metrics['timestamp']} by {metrics['user']}")
    logging.info("Per-class performance:")
    for class_name, class_metric in class_metrics.items():
        logging.info(f"{class_name}:")
        logging.info(f"  Precision: {class_metric['precision']:.4f}")
        logging.info(f"  Recall: {class_metric['recall']:.4f}")
        logging.info(f"  F1-score: {class_metric['f1']:.4f}")
    logging.info(f"Overall accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    logging.info(f"Weighted F1-score: {metrics['weighted_f1']:.4f}")
    
    return metrics


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        logging.info(f"Saved by user {checkpoint['user']} at {checkpoint['timestamp']}")
        logging.info(f"Validation metrics: {checkpoint['val_metrics']}")
        
        return checkpoint['epoch']
    
    except Exception as e:
        logging.error(f"Error loading checkpoint: {str(e)}")
        return None

def save_checkpoint(epoch, model, optimizer, scheduler, train_metrics, val_metrics, timestamp, user):
    os.makedirs('models', exist_ok=True)
    
    checkpoint_name = f'checkpoint_epoch_{epoch+1}_{timestamp.replace(" ", "_").replace(":", "-")}.pth'
    checkpoint_path = os.path.join('models', checkpoint_name)
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'timestamp': timestamp,
        'user': user
    }
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint to: {checkpoint_path}")
    
    best_model_path = os.path.join('models', 'best_model.pth')
    torch.save(checkpoint, best_model_path)
    logging.info(f"Updated best model: {best_model_path}")
    
    
def test_model(model, device, timestamp="2025", user="qwqwqweq"):
    transform = None
    base_path = '/data/Dataset_Bone/_train/mri'
    batch_size = 16
    num_workers = 4
    
    test_dataset_t1 = BrainMRIDataset(base_path, 'T1', transform=transform, mode='test')
    test_dataset_t2 = BrainMRIDataset(base_path, 'T2', transform=transform, mode='test')
    
    validate_test_datasets([test_dataset_t1, test_dataset_t2])
    logging.info(f"Test dataset sizes - T1: {len(test_dataset_t1)}, T2: {len(test_dataset_t2)}")
    
    test_loader_t1 = DataLoader(
        test_dataset_t1, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    test_loader_t2 = DataLoader(
        test_dataset_t2, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    model.eval()
    test_pred = []
    test_true = []
    test_probs = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    n_batches = min(len(test_loader_t1), len(test_loader_t2))
    logging.info(f"Starting evaluation on {n_batches} test batches")
    
    with torch.no_grad():
        for batch_idx, (t1_batch, t2_batch) in enumerate(zip(test_loader_t1, test_loader_t2)):
            if batch_idx >= n_batches:
                break
            
            if batch_idx % 10 == 0:
                logging.info(f"Processing test batch {batch_idx}/{n_batches}")
            
            t1_data, t1_labels = t1_batch
            t2_data, _ = t2_batch
            
            if t1_data.size(0) != t2_data.size(0):
                logging.warning(f"Skipping batch {batch_idx}: size mismatch")
                continue
            
            t1_data, t1_labels = t1_data.to(device), t1_labels.to(device)
            t2_data = t2_data.to(device)
            
            try:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(t1_data, t2_data)
                    loss = criterion(outputs, t1_labels)
                
                test_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                test_pred.extend(predicted.cpu().numpy())
                test_true.extend(t1_labels.cpu().numpy())
                test_probs.extend(probabilities.cpu().numpy())
                
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
    
    if len(test_pred) > 0:
        test_pred = [int(x) for x in test_pred]
        test_true = [int(x) for x in test_true]
        conf_matrix = confusion_matrix(test_true, test_pred).tolist()
        
        try:
            kappa = float(cohen_kappa_score(test_true, test_pred))
        except Exception as e:
            logging.error(f"Error calculating Cohen's Kappa: {str(e)}")
            kappa = float('nan')
            
        class_distribution = np.bincount(test_true, minlength=3).tolist()
        
        per_class_metrics = {}
        for i, class_name in enumerate(['Normal', 'Osteopenia', 'Osteoporosis']):
            per_class_metrics[class_name] = {
                'precision': float(precision_score(test_true, test_pred, labels=[i], average=None, zero_division=0)[0]),
                'recall': float(recall_score(test_true, test_pred, labels=[i], average=None, zero_division=0)[0]),
                'f1': float(f1_score(test_true, test_pred, labels=[i], average=None, zero_division=0)[0]),
                'support': int(np.sum(np.array(test_true) == i))
            }
        
        test_metrics = {
            'timestamp': str(timestamp),
            'user': str(user),
            'loss': float(test_loss / n_batches if n_batches > 0 else float('inf')),
            'accuracy': float(accuracy_score(test_true, test_pred)),
            'macro_precision': float(precision_score(test_true, test_pred, average='macro', zero_division=0)),
            'macro_recall': float(recall_score(test_true, test_pred, average='macro', zero_division=0)),
            'macro_f1': float(f1_score(test_true, test_pred, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(test_true, test_pred, average='weighted', zero_division=0)),
            'confusion_matrix': conf_matrix,
            'per_class_metrics': per_class_metrics,
            'class_distribution': class_distribution,
            'roc_auc': convert_to_serializable(calculate_multiclass_roc_auc(test_true, test_probs)),
            'cohen_kappa': kappa
        }
        
        log_test_results(test_metrics, ['Normal', 'Osteopenia', 'Osteoporosis'], test_true, np.array(conf_matrix))
        
        save_test_results(test_metrics, timestamp)
        
    else:
        test_metrics = create_empty_test_metrics()
        logging.error("No valid predictions were made during testing")
    
    return test_metrics

def calculate_multiclass_roc_auc(y_true, y_prob):
    timestamp = "2025"
    user = "qwqwqweq"
    try:
        y_true_onehot = label_binarize(y_true, classes=[0, 1, 2])
        
        auc_scores = {}
        for i in range(3):
            if len(np.unique(y_true_onehot[:, i])) == 2:
                try:
                    auc_scores[i] = roc_auc_score(y_true_onehot[:, i], np.array(y_prob)[:, i])
                except ValueError as e:
                    logging.warning(f"Unable to calculate ROC AUC for class {i}: {str(e)}")
                    auc_scores[i] = float('nan')
            else:
                auc_scores[i] = float('nan')
        
        valid_scores = [score for score in auc_scores.values() if not np.isnan(score)]
        mean_auc = np.mean(valid_scores) if valid_scores else float('nan')
        
        logging.info(f"ROC AUC calculation completed at {timestamp} by {user}")
        logging.info(f"Per-class ROC AUC scores: {auc_scores}")
        logging.info(f"Mean ROC AUC score: {mean_auc}")
        
        return {
            'per_class': auc_scores,
            'mean': mean_auc,
            'timestamp': timestamp,
            'user': user
        }
    except Exception as e:
        logging.error(f"Error calculating ROC AUC: {str(e)}")
        return {
            'per_class': {0: float('nan'), 1: float('nan'), 2: float('nan')},
            'mean': float('nan'),
            'timestamp': timestamp,
            'user': user
        }

def validate_test_datasets(datasets):
    timestamp = "2025"
    user = "qwqwqweq"
    expected_labels = {0, 1, 2}
    
    logging.info(f"Starting test dataset validation at {timestamp} by {user}")
    
    for dataset_idx, dataset in enumerate(datasets):
        labels = [label for _, label in dataset.samples]
        unique_labels = set(np.unique(labels))
        
        if not unique_labels.issubset(expected_labels):
            raise ValueError(
                f"Dataset {dataset_idx + 1} contains invalid labels: {unique_labels}\n"
                f"Expected labels should be in {expected_labels}"
            )
        
        class_counts = np.bincount(labels, minlength=3)
        class_names = ['Normal', 'Osteopenia', 'Osteoporosis']
        
        logging.info(f"Dataset {dataset_idx + 1} class distribution:")
        for i, (name, count) in enumerate(zip(class_names, class_counts)):
            logging.info(f"  {name}: {count} samples ({count/len(labels)*100:.2f}%)")
        
        if len(unique_labels) != 3:
            missing_labels = expected_labels - unique_labels
            logging.warning(
                f"Dataset {dataset_idx + 1} is missing classes: "
                f"{[class_names[l] for l in missing_labels]}"
            )

        min_samples = min(class_counts)
        max_samples = max(class_counts)
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        if imbalance_ratio > 3:
            logging.warning(
                f"Dataset {dataset_idx + 1} shows significant class imbalance "
                f"(imbalance ratio: {imbalance_ratio:.2f})"
            )
    
    logging.info(f"Test datasets validation completed successfully at {timestamp}")
    logging.info(f"Validation performed by user: {user}")
    return True

def log_test_results(metrics, class_names, test_true, conf_matrix):
    logging.info("\nTest Results:")
    logging.info(f"Number of test samples: {len(test_true)}")
    
    logging.info("\nClass Distribution:")
    for i, class_name in enumerate(class_names):
        count = metrics['class_distribution'][i]
        percentage = (count / len(test_true)) * 100
        logging.info(f"{class_name}: {count} ({percentage:.2f}%)")
    
    logging.info("\nConfusion Matrix:")
    logging.info("预测 →")
    logging.info("实际 ↓")
    for i, row in enumerate(conf_matrix):
        logging.info(f"{class_names[i]}: {row}")
    
    logging.info("\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['per_class_metrics'].items():
        logging.info(f"\n{class_name}:")
        logging.info(f"Precision: {class_metrics['precision']:.4f}")
        logging.info(f"Recall: {class_metrics['recall']:.4f}")
        logging.info(f"F1-score: {class_metrics['f1']:.4f}")
    
    logging.info("\nOverall Metrics:")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logging.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj

def save_test_results(metrics, timestamp):
    try:
        serializable_metrics = convert_to_serializable(metrics)
        
        results_file = f'test_results_{timestamp.replace(" ", "_").replace(":", "-")}.json'
        
        with open(results_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        logging.info(f"\nDetailed results saved to: {results_file}")
        
    except Exception as e:
        logging.error(f"Error saving test results: {str(e)}")
        raise

def create_empty_test_metrics():
    return {
        'timestamp': "2025",
        'user': "qwqwqweq",
        'loss': float('inf'),
        'accuracy': 0.0,
        'macro_precision': 0.0,
        'macro_recall': 0.0,
        'macro_f1': 0.0,
        'weighted_f1': 0.0,
        'confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        'per_class_metrics': {
            'Normal': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0},
            'Osteopenia': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0},
            'Osteoporosis': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}
        },
        'class_distribution': [0, 0, 0],
        'roc_auc': {'per_class': {0: 0.0, 1: 0.0, 2: 0.0}, 'mean': 0.0},
        'cohen_kappa': 0.0
    }

if __name__ == '__main__':
    try:
        timestamp = "2025"
        current_user = "qwqwqweq"
        model, best_val_metrics = train_model()
        logging.info(f"\nBest Validation Metrics: {best_val_metrics}")
        logging.info("\nStarting test evaluation...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_metrics = test_model(model, device, timestamp, current_user)
        
        final_results = {
            'timestamp': timestamp,
            'user': current_user,
            'best_validation_metrics': best_val_metrics,
            'test_metrics': test_metrics
        }
        
        results_file = f'models/final_results_{timestamp.replace(" ", "_").replace(":", "-")}.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=4)
        
        logging.info(f"\nFinal results saved to: {results_file}")
        
    except Exception as e:
        logging.exception("An error occurred during training or testing:")
        raise
