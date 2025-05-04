# DICFusion

The initial source code for the paper DICFusion (Under Review) has been uploaded.

## Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- PyTorch 1.8.0+
- See `requirements.txt` for detailed dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DICFusion.git
cd DICFusion
```

2. Create and activate conda environment:
```bash
conda create -n dicfusion python=3.8
conda activate dicfusion
```

3. Install dependencies:
```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Project Structure

```
DICFusion/
├── archs/          # Model architecture definitions
├── data/           # Data processing code
├── datasets/       # Datasets
├── experiments/    # Experiment configurations and results
├── models/         # Model implementations
├── options/        # Configuration files
└── utils/          # Utility functions
```

## Usage

### Testing

To test a pre-trained model:

1. **Prepare the test datasets**:
   - Ensure you have the testing data in the following directories:
     - `datasets/test/TNO/` (containing `ir` and `vi` folders)
     - `datasets/test/RoadScene/` (containing `ir` and `vi` folders)

2. **Check the pre-trained model**:
   - Make sure the pre-trained model is located at:
     ```
     experiments/Train_0503_DICFusegNet_dim64_B8_256_AdamW_tanh_None_Sobely1_MAF_CLIP0.1_1e-4/models/net_g_10000.pth
     ```
   - If the model is in a different location, update the path in the configuration file:
     ```yaml
     path:
       pretrain_network_g: /path/to/your/model.pth
     ```

3. **Run the test script**:
   ```bash
   python test.py -opt options/test/TCSVT/test_dicfuseg3.yml
   ```

4. **View results**:
   - The fused images will be saved in:
     ```
     results/Test_DICFusegNet/visualization/[dataset_name]/
     ```
   - The test progress will be displayed with a progress bar

### Training

The complete model implementation and detailed training procedures will be released after the paper is accepted.

## Citation

If you use our code or model, please cite our paper:

```bibtex
@article{author2023dicfusion,
    title={DICFusion: Deep Image Complementary Fusion},
    author={Author, A. and Author, B.},
    journal={Under Review},
    year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 