# Age, Gender, and Emotion Prediction Models

This repository contains the training code and implementation for age, gender, and emotion prediction models, specifically designed to run on NVIDIA Jetson Nano with a Raspberry Pi camera module for real-time applications. The training scripts are provided in Jupyter Notebook (.ipynb) format, and the live inference script is written in Python (.py).

## Project Overview

The project includes two separate models: one for predicting age and gender, and another for predicting emotions. These models were trained on distinct datasets to achieve the best performance in their respective tasks. The real-time application runs on NVIDIA Jetson Nano, utilizing a Raspberry Pi camera module to capture video streams for prediction.

### Age and Gender Prediction Model

- **Dataset**: The model was trained using the [UTKFace dataset](https://www.kaggle.com/code/eward96/age-and-gender-prediction-on-utkface), which is widely recognized for its diverse set of face images annotated with age and gender. This dataset provides a solid foundation for developing accurate age and gender prediction models.
- **Training Notebook**: Included in the repository is the Jupyter Notebook used for training the age and gender prediction model, outlining the model architecture, training process, and evaluation.

### Emotion Prediction Model

- **Dataset**: Training for the emotion prediction model was conducted using the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), which contains facial expressions categorized into several emotions. This dataset is instrumental in creating models capable of understanding and predicting human emotions based on facial expressions.
- **Training Notebook**: The repository also includes the Jupyter Notebook for the emotion prediction model's training, detailing the dataset preprocessing, model architecture, training strategy, and performance metrics.

## Real-time Application

The real-time application is designed to run on NVIDIA Jetson Nano, utilizing a Raspberry Pi camera module to capture live video feed. The script processes the video stream, applying the age, gender, and emotion prediction models to detect and predict these attributes in real-time.

## Setup and Installation

Instructions on setting up the NVIDIA Jetson Nano, connecting the Raspberry Pi camera, and running the real-time prediction script are provided. Ensure that all dependencies are installed, and the environment is properly configured before attempting to run the application.

## Usage

Details on how to use the real-time application, including command-line arguments and options, are outlined. This section provides clear instructions on starting the video stream and interacting with the prediction models.

## Contributing

We welcome contributions to this project! Whether it's improving the models, enhancing the real-time application, or fixing bugs, please feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the creators of the UTKFace and FER2013 datasets for providing the data necessary for training our models.
- This project would not have been possible without the support and resources provided by the NVIDIA Jetson Nano community and the developers of the Raspberry Pi camera module.
