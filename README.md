# MobileCLIP2 Fine-Tuning for Edge AI

This project measures how well Apple's [MobileCLIP2-S0](https://huggingface.co/apple/MobileCLIP2-S0), a small CLIP-style model built for mobile devices, classifies images. It also checks whether further training is worth it before the model is deployed to a smartphone.

The dataset is CIFAR-10. The notebook compares three approaches and then prepares the model for on-device inference.

## Results

| Approach | What it does | Accuracy |
|---|---|---|
| Zero-shot | No training. Each image is matched to text prompts such as "a photo of a cat." | 94.99% (test) |
| Linear probe | The model is frozen. A single linear layer is trained on its image embeddings. | 96.86% (validation) |
| Fine-tuning | The image encoder is retrained along with the classification head. | 96.84% (test) |

The pretrained model is already strong without any training. A linear probe adds about 2 points. Fine-tuning the whole image encoder adds nothing beyond that, so for this task the linear probe gives the best accuracy for its cost.

The fine-tuned model's per-class F1 scores range from 0.93 to 0.98. Cat and dog are the weakest classes.

## Setup

Requirements: Python 3.12 or 3.13, [Poetry](https://python-poetry.org/), and a CUDA GPU. The fine-tuning code uses CUDA mixed precision, so it won't run on a CPU-only machine as written. The original run used a Tesla T4.

```bash
poetry install
poetry run python -m ipykernel install --user --name mobileclip2
```

Open `MobileCLIP2 FineTuning EdgeAI.ipynb` and select the `mobileclip2` kernel. The first cell installs packages with `!pip`. Skip it if you used Poetry.

The first run downloads:
- the MobileCLIP2-S0 checkpoint (about 300 MB) from Hugging Face. Setting `HF_TOKEN` speeds this up.
- CIFAR-10 (about 170 MB) into `./data`.

## Notebook walkthrough

1. **Load the model.** Download the MobileCLIP2-S0 weights and load them with `open_clip`. S0 expects pixel values in [0, 1] with no mean/std normalization.
2. **Prepare the data.** CIFAR-10 images (32×32) are upscaled to the model's 256×256 input size. The 50,000 training images are split into 45,000 for training and 5,000 for validation, with 500 per class and a fixed seed. Training images get random-crop and horizontal-flip augmentation.
3. **Zero-shot baseline.** Encode one text prompt per class and assign each image to the most similar prompt.
4. **Experiment A: linear probe.** Freeze the model, extract a 512-dimensional normalized embedding for every image, and train `nn.Linear(512, 10)` for 10 epochs.
5. **Experiment B: fine-tuning.** Unfreeze the image encoder; the text encoder stays frozen. Train with AdamW using separate learning rates (1e-5 for the encoder, 5e-4 for the head) and fp16 mixed precision. Keep the checkpoint with the best validation accuracy. The head starts from the weights trained in Experiment A.
6. **Evaluate.** Report test accuracy, macro precision, recall and F1, and a per-class report, then plot the comparison.
7. **Prepare for deployment.** Reparameterize the MobileOne/FastViT blocks, merging their training-time branches into single convolutions for faster inference. Wrap the encoder and classifier into one module and export it with TorchScript.
8. **Benchmark.** Measure single-image latency and checkpoint size.

## Outputs

| File | Contents |
|---|---|
| `mobileclip2_finetuned_cifar10.pth` | Fine-tuned CLIP weights, the classifier head, and the class names (about 286 MB) |
| `mobileclip2_android.pt` | TorchScript export of the reparameterized image classifier |

## Known limitations

- **Mismatched features in fine-tuning.** During fine-tuning, training and validation feed raw image features to the classifier. Test evaluation and the exported model normalize those features first. The reported fine-tuned test accuracy is therefore not measured on exactly the setup that was trained.
- **Training may not have finished.** The saved output of the fine-tuning cell stops after epoch 4 of 5, so the best-checkpoint restore may not have run.
- **Mixed evaluation splits.** The linear-probe accuracy is measured on validation data, while the other two are on test data.
- **Very small batch size.** `BATCH_SIZE = 4` makes training slow and gives noisy BatchNorm statistics.
- **Export not ready for phones yet.** The model is traced on CUDA and includes the unused text encoder. For a phone, trace on CPU using only the image encoder, and consider ExecuTorch (Android) or Core ML (iOS) instead of TorchScript.
- **Benchmarks don't reflect a phone.** Latency (about 9 ms) was measured on a T4 GPU, and the 286 MB size is the full fp32 checkpoint rather than the exported model.

## Acknowledgements

- [MobileCLIP / MobileCLIP2](https://github.com/apple/ml-mobileclip) by Apple
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) by Krizhevsky, Nair and Hinton
