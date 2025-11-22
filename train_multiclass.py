import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.CNN_TUMOR_MULTICLASS import TumorClassifier
from utils_multilclass.data_loader import get_dataloaders


def main():
    # Device and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[DEBUG] Using device: {device}')

    train_loader, val_loader = get_dataloaders()
    print(f'[DEBUG] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

    # Model, loss, optimizer
    model = TumorClassifier(num_classes=3, input_size=224)
    model = model.to(device)
    print(f'[DEBUG] Model moved to device. Model summary (repr):')
    print(repr(model))

    criterion = nn.CrossEntropyLoss()

    # START WITH LOWER LEARNING RATE to prevent spike
    base_lr = 0.0001  # Reduced from 0.001
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)

    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Warmup configuration
    warmup_epochs = 5  # Gradually increase LR for first 5 epochs
    warmup_start_lr = base_lr
    warmup_target_lr = 0.001  # Target LR after warmup

    # Initialize lists to store training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    # Early stopping parameters
    early_stopping_patience = 10
    early_stopping_counter = 0

    # Training loop
    num_epochs = 50  # Increased from 20
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    print(f'[DEBUG] Starting training for {num_epochs} epochs')
    print(f'[DEBUG] Using LR warmup: {warmup_start_lr:.6f} -> {warmup_target_lr:.6f} over {warmup_epochs} epochs')
    print(f'[DEBUG] Gradient clipping enabled with max_norm=1.0')

    for epoch in range(num_epochs):
        # Learning rate warmup for first few epochs
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            current_lr = warmup_start_lr + (warmup_target_lr - warmup_start_lr) * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f'[DEBUG] Epoch {epoch + 1}/{num_epochs} - WARMUP LR: {current_lr:.6f}')
        else:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'[DEBUG] Epoch {epoch + 1}/{num_epochs} - LR: {current_lr:.6f}')

        learning_rates.append(current_lr)

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # GRADIENT CLIPPING to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Periodic debug message per batch
            if batch_idx % 10 == 0 or batch_idx == 1:
                print(f'[DEBUG] Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} - loss: {loss.item():.4f}, inputs.shape: {tuple(inputs.shape)}')

        train_loss /= len(train_loader) if len(train_loader) > 0 else 1.0
        train_accuracy = correct / total if total > 0 else 0.0
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader) if len(val_loader) > 0 else 1.0
        val_accuracy = correct / total if total > 0 else 0.0
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Only use scheduler after warmup period
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr:.6f}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2%}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2%}')

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model_multiclass.pth')
            print(f'[DEBUG] New best model saved with val_accuracy: {best_val_accuracy:.2%}, val_loss: {best_val_loss:.4f}')
        else:
            early_stopping_counter += 1
            print(f'[DEBUG] No improvement. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}')

        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f'[DEBUG] Early stopping triggered after {epoch + 1} epochs')
            break

    accuracy = correct / total if total > 0 else 0.0
    print(f'\n✅ Training completed!')
    print(f'Best Validation Accuracy: {best_val_accuracy:.2%}')
    print(f'Best Validation Loss: {best_val_loss:.4f}')

    # Visualize training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    axes[0, 0].plot(train_losses, label='Training Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0, 0].axvline(x=warmup_epochs, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Loss History')
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[0, 1].plot(train_accuracies, label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(val_accuracies, label='Validation Accuracy', linewidth=2)
    axes[0, 1].axvline(x=warmup_epochs, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].set_title('Accuracy History')
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate plot
    axes[1, 0].plot(learning_rates, linewidth=2, color='green')
    axes[1, 0].axvline(x=warmup_epochs, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule (with Warmup)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Overfitting gap
    accuracy_gap = [t - v for t, v in zip(train_accuracies, val_accuracies)]
    axes[1, 1].plot(accuracy_gap, linewidth=2, color='orange')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].axvline(x=warmup_epochs, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train Acc - Val Acc')
    axes[1, 1].set_title('Overfitting Gap (closer to 0 is better)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print('[DEBUG] Training history saved to training_history.png')
    plt.show()


if __name__ == "__main__":
    main()
