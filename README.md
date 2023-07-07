# U-Net Architecture for Image Segmentation

This repository contains code that implements the U-Net architecture for image segmentation from scratch. The U-Net architecture is a popular and effective deep learning model for various image segmentation tasks.

## Implementation Details

The code includes the following components:

- **Model Implementation**: The U-Net architecture is implemented using the `UNET` class. It consists of a down part and an up part. The down part consists of multiple double convolutional layers followed by max-pooling. The up part consists of transpose convolutional layers and skip connections. The final output is obtained through a convolutional layer.

- **Dataset Preparation**: The `Dataset` class is responsible for loading the images and masks for training and validation. The images and masks are preprocessed and transformed using the specified transformations.

- **Training and Evaluation**: The `train` function is used to train the U-Net model. It performs forward and backward propagation, updates the weights, and tracks the training progress. The `check_accuracy` function evaluates the model's accuracy and dice score on the validation dataset.

- **Hyperparameters and Configurations**: The hyperparameters such as learning rate, batch size, number of epochs, and image dimensions are specified. The model is trained using the Adam optimizer and the binary cross-entropy loss function.

## Usage

To use the code, follow these steps:

1. Prepare your dataset: Organize your dataset into separate directories for images and corresponding masks.

2. Configure the hyperparameters: Modify the hyperparameters such as learning rate, batch size, number of epochs, and image dimensions according to your requirements.

3. Specify the dataset directories: Update the `train_img_dir`, `train_mask_dir`, `val_img_dir`, and `val_mask_dir` variables with the appropriate paths to your dataset directories.

4. Run the code: Execute the code in a Python environment that has the required dependencies installed. The code will train the U-Net model on the specified dataset and display the training progress.

## Dependencies

The following dependencies are required to run the code:

- PyTorch
- torchvision
- numpy
- PIL
- albumentations
- tqdm

Install the dependencies using the following command:
```script
pip install torch torchvision numpy Pillow albumentations tqdm
```


## Acknowledgments

The code in this repository is inspired by the U-Net architecture introduced in the paper:

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
