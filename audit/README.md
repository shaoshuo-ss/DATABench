## Dataset Auditing Interface

The auditing interface is defined in dataset_audit.py and provides a standardized way to implement new dataset auditing algorithms.

### Interface Definition

All auditing methods must inherit from the base `DatasetAudit` class:

```python
class DatasetAudit:
    def __init__(self, params: dict):
        """Initialize the auditing method with parameters"""
        pass
    
    def process_dataset(self, ori_dataset, aux_dataset=None):
        """
        Embed a watermark into the original dataset.

        Args:
            ori_dataset: The original dataset.
            params (dict): Additional parameters for processing.

        Returns:
         A tuple containing:
                - pub_dataset: The processed dataset with embedded watermark.
                - aux (dict): Auxiliary data required for verification.
        """
        pass
    
    def verify(self, pub_dataset, model, aux: dict, aux_dataset=None):
        """
        Main verification function for dataset auditing
        
        Args:
            pub_dataset: The public dataset to audit
            model: The target model
            aux: Auxiliary information dictionary
            aux_dataset: Optional auxiliary dataset
            
        Returns:
            Auditing result (typically a confidence score)
        """
        pass
```

## Available Auditing Methods

**Internal-feature-based auditing**:
- `MIA`: from the paper "Membership inference attacks against machine learning models" (Oakland 2017)
- `Rapid`: from the paper "Is difficulty calibration all we need? towards more practical membership inference attacks" (ACM CCS 2024)
- `Dataset Inference (DI)`: from the paper "Dataset inference: Ownership resolution in machine learning" (ICLR 2020)
- `Data-use Auditing (DUA)`: from the paper "A general framework for datause auditing of ml models" (ACM CCS 2024)

**External-feature-based auditing**:
- `DVBW`: from the paper "Black-box dataset ownership verification via backdoor watermarking" (TIFS 2023)
- `Untargeted Backdoor Watermark (UBW-P/C)`: from the paper "Untargeted backdoor watermark: Towards harmless and stealthy dataset copyright protection" (NeurIPS 2022)
- `ZeroMark`: from the paper "Zeromark: Towards dataset ownership verification without disclosing watermark" (NeurIPS 2024)
- `Domain Watermark (DW)`: from the paper "Domain watermark: Effective and harmless dataset copyright protection is closed at hand" (NeurIPS 2023)

Note: To implement DW, you may need to download the file of domain data from the [official repository](https://github.com/JunfengGo/Domain-Watermark) or generate your own domain data using the provided code in the `Learning_to_diversify` directory.

## Adding New Auditing Methods

1. Create a new file in audit directory
2. Inherit from the `DatasetAudit` base class
3. Implement the `process_dataset` and `verify` method
4. Add configuration files in `config/audit/`