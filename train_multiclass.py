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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Initialize lists to store training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Early stopping parameters
    early_stopping_patience = 10
    early_stopping_counter = 0

    # Training loop
    num_epochs = 50  # Increased from 20
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    print(f'[DEBUG] Starting training for {num_epochs} epochs with early stopping')

    for epoch in range(num_epochs):
        print(f'[DEBUG] Epoch {epoch + 1}/{num_epochs} - starting')
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

        # Step the learning rate scheduler
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2%}, '
              f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}')

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
    print(f'Validation Accuracy: {accuracy:.2%}')

    # Visualize training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy History')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
