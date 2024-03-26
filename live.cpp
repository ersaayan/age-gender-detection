#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/tools/gen_op_registration.h>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace tflite;

const string faceCascadePath = "haarcascades_models/haarcascade_frontalface_default.xml";
const string ageModelPath = "./pre-trained-models/age_detection_model_100epochs_opt.tflite";
const string genderModelPath = "./pre-trained-models/gender_detection_model_100epochs_opt.tflite";
vector<string> genderLabels = {"Male", "Female"};

unique_ptr<Interpreter> buildInterpreter(const string& modelPath) {
    auto model = FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model) {
        cerr << "Failed to load model: " << modelPath << endl;
        exit(1);
    }
    ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder builder(*model, resolver);
    unique_ptr<Interpreter> interpreter;
    builder(&interpreter);
    interpreter->AllocateTensors();
    return interpreter;
}

void setTensor(unique_ptr<Interpreter>& interpreter, int index, Mat& img) {
    float* tensorData = interpreter->typed_tensor<float>(index);
    Mat tempImg;
    img.convertTo(tempImg, CV_32F);
    memcpy(tensorData, tempImg.data, tempImg.total() * tempImg.elemSize());
}

int getGenderPrediction(unique_ptr<Interpreter>& interpreter) {
    TfLiteTensor* outputTensor = interpreter->tensor(interpreter->outputs()[0]);
    float* scores = outputTensor->data.f;
    return scores[0] >= 0.5 ? 1 : 0;
}

int getAgePrediction(unique_ptr<Interpreter>& interpreter) {
    TfLiteTensor* outputTensor = interpreter->tensor(interpreter->outputs()[0]);
    return static_cast<int>(round(outputTensor->data.f[0]));
}

int main() {
    CascadeClassifier faceClassifier(faceCascadePath);
    if (faceClassifier.empty()) {
        cerr << "Error loading face cascade" << endl;
        return -1;
    }
    auto ageInterpreter = buildInterpreter(ageModelPath);
    auto genderInterpreter = buildInterpreter(genderModelPath);

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error opening video stream" << endl;
        return -1;
    }

    Mat frame, gray;
    vector<Rect> faces;
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        faceClassifier.detectMultiScale(gray, faces);

        auto start = chrono::high_resolution_clock::now();

        for (auto& face : faces) {
            rectangle(frame, face, Scalar(255, 0, 0), 2);
            Mat roiGray = gray(face);
            Mat roiColor = frame(face).clone();
            resize(roiGray, roiGray, Size(48, 48));
            resize(roiColor, roiColor, Size(200, 200));

            // Set the input tensor for gender
            setTensor(genderInterpreter, genderInterpreter->inputs()[0], roiColor);
            genderInterpreter->Invoke();
            int genderIndex = getGenderPrediction(genderInterpreter);
            string genderLabel = genderLabels[genderIndex];
            putText(frame, genderLabel, Point(face.x, face.y + face.height + 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

            // Set the input tensor for age
            setTensor(ageInterpreter, ageInterpreter->inputs()[0], roiColor);
            ageInterpreter->Invoke();
            int age = getAgePrediction(ageInterpreter);
            putText(frame, "Age=" + to_string(age), Point(face.x, face.y + face.height + 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Total time=" << elapsed.count() << "s" << endl;

        imshow("Detector", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
