import torch
import torch.nn as nn


class CLIPWithLinearProbeStandard(nn.Module):
    def __init__(self, clip_model, num_classes, dropout=0.5):
        super().__init__()
        self.clip = clip_model

        # Freeze backbone
        for param in self.clip.parameters():
            param.requires_grad = False

        # Get feature dimension
        dummy = torch.randn(1, 3, 224, 224)  # adjust input size if needed
        feature_dim = self.clip.visual(dummy).shape[1]
        print(f"Visual feature dimension: {feature_dim}")

        # Linear head (trainable)
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(feature_dim, num_classes))
        self.linear_head = nn.Sequential(*layers)  # keep it separate

    def forward(self, image=None, text=None):
        # Get frozen features from visual backbone
        features = self.clip.visual(image)  # [B, feature_dim]
        # Pass through trainable linear head
        logits = self.linear_head(features)
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
import numpy as np
from torch.utils.data import DataLoader

# ----------------------------
# Wrapper for penultimate features
# ----------------------------
class CLIPVisualPenultimate(nn.Module):
    """
    Wrap CLIP visual backbone to return penultimate features (before projection)
    """
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.proj = self.visual.proj  # save original projection
        self.visual.proj = None       # temporarily disable projection

    def forward(self, x):
        feats = self.visual(x)  # [B, feature_dim], no projection
        return feats

# ----------------------------
# Linear probe class
# ----------------------------
class CLIPWithLinearProbeExact(nn.Module):
    def __init__(self, clip_model, num_classes, device="cuda"):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.clip_visual = CLIPVisualPenultimate(clip_model).eval().to(device)
        self.linear = None  # initialized later when feature dim is known

        # Freeze CLIP backbone
        for param in self.clip_visual.parameters():
            param.requires_grad = False

    # ----------------------------
    # Extract features from a dataloader
    # ----------------------------
    def extract_features(self, dataloader):
        feats, labels = [], []
        with torch.no_grad():
            for images, y in dataloader:
                images = images.to(self.device)
                f = self.clip_visual(images)
                feats.append(f.cpu())
                labels.append(y)
        return torch.cat(feats), torch.cat(labels)

    # ----------------------------
    # Train logistic regression once for a given weight decay
    # ----------------------------
    def _train_once(self, X, y, lmbd, max_iter=1000):
        n, d = X.shape
        if self.linear is None:
            self.linear = nn.Linear(d, self.num_classes, bias=True).to(self.device)

        model = nn.Linear(d, self.num_classes, bias=True).to(self.device)
        X, y = X.to(self.device), y.to(self.device)

        optimizer = LBFGS(model.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe",
                          weight_decay=lmbd)

        def closure():
            optimizer.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            return loss

        optimizer.step(closure)

        # compute accuracy
        with torch.no_grad():
            preds = model(X).argmax(dim=1)
            acc = (preds == y).float().mean().item()

        return model, acc

    # ----------------------------
    # Fit logistic regression and sweep weight decay (λ)
    # ----------------------------
    def fit(self, train_loader, val_loader, max_iter=1000):
        X_train, y_train = self.extract_features(train_loader)
        X_val, y_val = self.extract_features(val_loader)

        lambdas = [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]
        best_acc, best_model, best_lambda = -1, None, None

        def evaluate_lambda(lmbd):
            model, _ = self._train_once(X_train, y_train, lmbd, max_iter)
            with torch.no_grad():
                logits = model(X_val.to(self.device))
                preds = logits.argmax(dim=1).cpu()
                acc = (preds == y_val).float().mean().item()
            return acc, model

        # simple parametric refinement loop
        while True:
            results = []
            for lmbd in lambdas:
                acc, model = evaluate_lambda(lmbd)
                results.append((acc, model, lmbd))

            accs = [r[0] for r in results]
            best_idx = int(np.argmax(accs))
            best_acc, best_model, best_lambda = results[best_idx]

            if len(lambdas) > 96:  # stopping criterion
                break

            # refine around best lambda
            if best_idx == 0:
                new_range = np.logspace(np.log10(lambdas[0]) - 2, np.log10(lambdas[0]), 8)
            elif best_idx == len(lambdas) - 1:
                new_range = np.logspace(np.log10(lambdas[-1]), np.log10(lambdas[-1]) + 2, 8)
            else:
                left, right = lambdas[best_idx - 1], lambdas[best_idx + 1]
                new_range = np.logspace(np.log10(left), np.log10(right), 8)

            lambdas = sorted(set(lambdas + list(new_range)))

        self.linear = best_model
        print(f"Best λ = {best_lambda:.2e}, Val Acc = {best_acc:.4f}")
        return best_acc

    # ----------------------------
    # Predict with logging
    # ----------------------------
    def predict(self, dataloader):
        X_list, y_list = [], []
        total_batches = len(dataloader)
        print(f"Starting prediction on {total_batches} batches...")

        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader, 1):
                images, labels = images.to(self.device), labels.to(self.device)
                feats = self.clip_visual(images)
                logits = self.linear(feats)
                preds = logits.argmax(dim=1)

                X_list.append(preds.cpu())
                y_list.append(labels.cpu())

                if i % 10 == 0 or i == total_batches:
                    print(f"Batch {i}/{total_batches} completed")

        all_preds = torch.cat(X_list)
        all_labels = torch.cat(y_list)
        acc = (all_preds == all_labels).float().mean().item()
        print(f"Prediction completed. Overall accuracy: {acc:.4f}")
        return all_preds, all_labels
