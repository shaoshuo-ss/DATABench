## Attack Interface

The attack interface is defined in attack_interface.py and provides a three-stage attack pipeline.

### Three-Stage Attack Pipeline

DATABench implements attacks through three distinct stages:

1. **Preprocessing Stage**: Data manipulation before training
2. **Training Stage**: Model training with specific techniques
3. **Postprocessing Stage**: Output manipulation after training

Each stage has its own interface:

```python
class Preprocessing:
    def __init__(self, args):
        self.args = args
    
    def process(self, dataset):
        """Process the dataset before training"""
        return processed_dataset

class Training:
    def __init__(self, args):
        self.args = args
    
    def train(self, model, dataset):
        """Train the model with specific techniques"""
        return trained_model

class Postprocessing:
    def __init__(self, args):
        self.args = args
    
    def wrap_model(self, model, aux_dataset=None):
        """Wrap the model with postprocessing techniques"""
        return wrapped_model
```

### Attack Configuration

Attacks are configured and instantiated in attack.py. The `get_attack` function demonstrates how to combine different stages. If you do not want to attack in other stages, you can simply use the default implementations of the other stages (i.e., `Preprocessing(), Training(), Postprocessing()`). Examples are as follows:

```python
def get_attack(args):
    if args.attack_method == "medianfilter":
        preprocessing = MedianFilterPreprocessing(args)
        training = Training(args)
        postprocessing = Postprocessing(args)
    elif args.attack_method == "dpsgd":
        preprocessing = Preprocessing(args)
        training = DPSGD(args)
        postprocessing = Postprocessing(args)
    # ... more attacks
```

## Available Attacks

### Evasion Attacks

**Preprocessing Attacks**:
- `Data Synthesis`: Data synthesis techniques, using generative models to create synthesis data instead of directly using the original dataset.
- `MedianFilterPreprocessing`: Median filtering.
- `GaussianFilterPreprocessing`: Gaussian filtering.
- `WaveletFilterPreprocessing`: Wavelet filtering.
- `AutoencoderPreprocessing`: Autoencoder-based denoising.

**Training Attacks**:
- `DPSGD`: Differential privacy during training
- `AdversarialTraining`: Adversarial training techniques
- `ASD`: Adaptive Splitting Poisoned Dataset (from [link](https://github.com/KuofengGao/ASD))

**Postprocessing Attacks**:
- `OutputNoise`: Injecting noise into the model's output.
- `FeatureNoise`: Injecting noise into the intermediate features.
- `RandomizedSmoothing`: Randomized smoothing defense.
- `Reprogramming`: Model reprogramming.
- `SCALE_UP`: Backdoor detection named SCALE_UP (from [link](https://github.com/JunfengGo/SCALE-UP))
- `SVMOutlierDetection`: Outlier detection using SVM.
- `KNNOutlierDetection`: Outlier detection using KNN.

### Forgery Attacks

In forgery attacks, we implement five different adversarial attacks method from `torchattacks`:
- `FGSM`
- `PGD`
- `UAP`
- `VNIFGSM`
- `TIFGSM`

## Adding New Attack Methods

1. Choose the appropriate stage(s) for your attack
2. Create new classes inheriting from `Preprocessing`, `Training`, or `Postprocessing`
3. Place them in the corresponding subdirectories under attack
4. Register your attack in attack.py
5. Add configuration files in `config/attack/`