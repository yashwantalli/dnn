# CIFAR-10H Human Uncertainty Modeling

## Objective
Model human uncertainty in image classification using the CIFAR-10H dataset.

## Approach
- CNN model
- KL Divergence loss
- Entropy-based uncertainty analysis

## Dataset
Download from:
https://github.com/jcpeterson/cifar-10h

Place `cifar10h-probs.npy` in project folder.

## Results
- Test KL Loss: ~1.06
- Entropy Correlation: ~0.18

## Conclusion
Model captures class distributions but has limited ability to fully model human uncertainty.