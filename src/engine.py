import torch
import torchvision
import torchvision.transforms.functional as F

def eval(model, preprocess, text,  test_loader, log_interval=10):
  with torch.no_grad(), torch.autocast("cuda"):
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch_idx, (samples, targets) in enumerate(test_loader):
      image_features = model.encode_image(samples)
      text_features = model.encode_text(text)
      image_features /= image_features.norm(dim=-1, keepdim=True)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
      preds = text_probs.argmax(dim=1)
      total_correct += (preds == targets).sum().item()
      total_samples += targets.size(0)
      if batch_idx % log_interval == 0 or batch_idx == len(test_loader):
        batch_acc = (preds == targets).float().mean().item() * 100
        print(f"[Batch {batch_idx:3d}/{len(test_loader)}] "
              f"Batch Acc: {batch_acc:6.2f}%")
  accuracy = total_correct / total_samples * 100
  print("=" * 50)
  print(f"📊 Evaluation Results")
  print(f"   • Accuracy : {accuracy:.2f}% ({total_correct}/{total_samples})")
  print("=" * 50)

def linear_probe_train(model, preprocess, text, train_loader, val_loader,   optimizer, criterion, epoch=1, total_epochs=10, log_interval=10, device='cuda'):
    """
    Train and validate a model for one epoch with batch and epoch-level logging.

    Args:
        model: torch.nn.Module
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        criterion: loss function (e.g., nn.CrossEntropyLoss())
        optimizer: optimizer (e.g., Adam)
        device: "cuda" or "cpu"
        epoch: current epoch number
        total_epochs: total number of epochs
        log_interval: log every N batches
    """

    # ---------------- Train ----------------
    model.train()
    train_loss, train_correct, train_samples = 0.0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
      inputs, targets = inputs.to(device), targets.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

      train_loss += loss.item() * inputs.size(0)
      preds = outputs.argmax(dim=1)
      train_correct += (preds == targets).sum().item()
      train_samples += targets.size(0)

      if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
        batch_acc = (preds == targets).float().mean().item() * 100
        print(f"[Epoch {epoch}/{total_epochs}] [Train Batch {batch_idx}/{len(train_loader)}] "
              f"Loss: {loss.item():.4f} | Batch Acc: {batch_acc:6.2f}%")

    train_avg_loss = train_loss / train_samples
    train_acc = train_correct / train_samples * 100

    print("-" * 60)
    print(f"📈 Epoch {epoch}/{total_epochs} Train Summary")
    print(f"   • Avg Loss : {train_avg_loss:.4f}")
    print(f"   • Accuracy : {train_acc:.2f}% ({train_correct}/{train_samples})")
    print("-" * 60)

    # ---------------- Validation ----------------
    model.eval()
    val_loss, val_correct, val_samples = 0.0, 0, 0

    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(val_loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        val_correct += (preds == targets).sum().item()
        val_samples += targets.size(0)

        if batch_idx % log_interval == 0 or batch_idx == len(val_loader):
          batch_acc = (preds == targets).float().mean().item() * 100
          print(f"[Epoch {epoch}/{total_epochs}] [Val Batch {batch_idx}/{len(val_loader)}] "
                f"Loss: {loss.item():.4f} | Batch Acc: {batch_acc:6.2f}%")

    val_avg_loss = val_loss / val_samples
    val_acc = val_correct / val_samples * 100

    print("=" * 60)
    print(f"📊 Epoch {epoch}/{total_epochs} Validation Summary")
    print(f"   • Avg Loss : {val_avg_loss:.4f}")
    print(f"   • Accuracy : {val_acc:.2f}% ({val_correct}/{val_samples})")
    print("=" * 60)

    return (train_avg_loss, train_acc), (val_avg_loss, val_acc)

