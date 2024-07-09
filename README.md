# Eye-Detection-using-CNN

Image classification is a core task in computer vision, requiring accurate image labeling. Deep convolutional neural networks (CNNs) excel at this by learning complex features from raw data. However, effective CNN design is challenging due to dataset complexity, architecture choice, and hyperparameter tuning.

This project employs advanced techniques like transfer learning with fine-tuning of pre-trained models and data augmentation (random cropping, flipping, rotations) to improve model robustness and training data diversity. It explores CNN architectures, optimization strategies, and evaluates trade-offs in model complexity, training time, and accuracy, benchmarking against traditional machine learning and pre-trained models.

## Dataset
The dataset used for training and evaluation is available on Kaggle: [Eye Image Dataset](https://www.kaggle.com/datasets/kayvanshah/eye-dataset/code). It includes images organized into the following categories:

- Close Look
- Forward Look
- Left Look
- Right Look

## Requirements
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Eye-Detection-using-CNN.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Eye-Detection-using-CNN
    ```

## Usage
1. Download the dataset from [Kaggle - Eye Image Dataset](https://www.kaggle.com/datasets/kayvanshah/eye-dataset/code) and place it in the project directory.
2. Open the Jupyter notebook:
    ```bash
    jupyter notebook EYE.ipynb
    ```
3. Run the cells in the notebook to train and evaluate the CNN model.

## Results
The high accuracy rate of 98.17% achieved on the eye detection task is a result of meticulous model development, including data preprocessing, hyperparameter tuning, regularization, transfer learning, data balancing, cross-validation, and ensemble learning. The use of cutting-edge architectures, effective optimization strategies, and ensemble techniques significantly contributed to the model's robust performance in localizing and classifying eye regions.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

