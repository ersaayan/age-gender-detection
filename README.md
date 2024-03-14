
# Age, Gender, and Emotion Detection

This repository hosts a collection of models and scripts designed for age, gender, and emotion detection, utilizing deep learning. The models are trained on the UTKFace dataset for age and gender prediction, and on the FER2013 dataset for emotion detection. The project is set up to run live tests with a Raspberry Pi camera on an Nvidia Jetson Nano, showcasing real-time detection capabilities.

## Datasets Used

- **Age and Gender Detection**: [UTKFace dataset](https://www.kaggle.com/code/eward96/age-and-gender-prediction-on-utkface)
- **Emotion Detection**: [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)

## Notebooks

The training code for the models is provided in Jupyter Notebooks:

- `keras_to_tflite.ipynb`: Converts Keras models to TensorFlow Lite format for efficient deployment on edge devices.
- `train_age_gender.ipynb`: Training script for the age and gender detection model.
- `train_emotion.ipynb`: Training script for the emotion detection model.

## Live Testing Script

- `live.py`: A Python script for running live tests with the Raspberry Pi camera on an Nvidia Jetson Nano. This script utilizes the TensorFlow Lite models for real-time detection.

## Setup and Usage

### Installation
```bash
sudo apt-get update && sudo apt-get upgrade -y
```
Install Python and Pip
```bash
sudo apt-get install python3 python3-pip
```
#### Install Dependencies
```bash
pip3 install scikit-build
```
```bash
pip3 install opencv-contrib-python
```

Now, you need to install tflite interpreter.

You do not need full tensorflow to just run the tflite interpreter.
The package tflite_runtime only contains the Interpreter class which is what we need.
It can be accessed by tflite_runtime.interpreter.Interpreter.
To install the tflite_runtime package, just download the Python wheel
that is suitable for the Python version running on your Jetson.

Check Python version

```bash
python3 --version
```
Download the appropriate version of python from [Google Coral Tflite](https://github.com/google-coral/pycoral/releases/).

for Python 3.6, download: tflite_runtime-2.5.0.post1-cp36-cp36m-linux_x86_64.whl (This is what I used)

### Training

The models were initially trained using Google Colab Pro with an A100 GPU for optimal performance. To retrain the models or train new ones, you can follow the steps outlined in the Jupyter Notebooks. Make sure to have a Google Colab Pro account for access to A100 runtime.

### Conversion to TensorFlow Lite

The `keras_to_tflite.ipynb` notebook includes detailed steps for converting the trained Keras models to TensorFlow Lite format. This conversion is crucial for deploying the models on edge devices like the Nvidia Jetson Nano.

### Running on Nvidia Jetson Nano

To run the models on an Nvidia Jetson Nano, you'll need to set up TensorFlow Lite. Follow these steps for installation:

1. Ensure that your Jetson Nano is updated and running the latest version of its operating system.
2. Install TensorFlow Lite by following the official TensorFlow guide for ARM64-based devices. This may involve downloading the TensorFlow Lite runtime and setting up the environment accordingly.

### Live Detection

To start live detection, ensure your Raspberry Pi camera is correctly connected to your Nvidia Jetson Nano. Then, execute the `live.py` script:

```bash
python live.py
```

This will activate the camera and start the age, gender, and emotion detection in real-time.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Age and Gender Prediction Model trained on [UTKFace dataset](https://www.kaggle.com/code/eward96/age-and-gender-prediction-on-utkface).
- Emotion Detection Model trained on [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).
- Models trained using Google Colab Pro's A100 runtime for optimal performance.

---

For more information, visit the project repository: [Age, Gender, and Emotion Detection](https://github.com/ersaayan/age-gender-detection/tree/main).
