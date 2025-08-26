import torch
import torchvision

def eval(model, preprocess, text,  test_loader):
  with torch.no_grad(), torch.autocast("cuda"):
    for data_iter_step, (samples, targets) in enumerate(test_loader):
      image = preprocess(torchvision.transforms.ToPILImage()(samples[0])).unsqueeze(0)
      image_features = model.encode_image(image)
      text_features = model.encode_text(text)
      image_features /= image_features.norm(dim=-1, keepdim=True)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
      print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

def linear_probe_train(model, preprocess, text, train_loader):
  for data_iter_step, (samples, targets) in enumerate(train_loader):
    print(samples.shape)
    print(targets)
    image = preprocess(torchvision.transforms.ToPILImage()(samples[0])).unsqueeze(0)
    print(preprocess(image).unsqueeze(0).shape)
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
    break

