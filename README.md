# Food Image Classifier App 🍔🍎

This project is a Python application designed to classify images as either food or non-food using the Food 5K dataset. The model leverages deep learning techniques to achieve high classification accuracy.

## Features

- **Deep Learning Model**: Built and trained a convolutional neural network (CNN) for image classification.
- **Data Augmentation**: Implemented data augmentation to improve model generalization.
- **Testing and Validation**: Conducted extensive testing and validation to ensure accuracy and robustness.
- **High Accuracy**: Achieved high classification accuracy through iterative model improvements.

## Dataset

The project uses the [Food 5K dataset](https://www.kaggle.com/datasets/trolukovich/food5k-image-dataset), which contains 5000 images split into three sets: training, validation, and testing. Each set contains images labeled as either food or non-food.

## Model Architecture

The Food or Not Food Keras model is a Convolutional Neural Network (CNN) designed to classify images as either containing food or not. The model accepts images of shape (224, 224, 3) and outputs a binary classification. Leveraging a pre-trained feature extractor Efficient-Net-V2, the model is trained on a diverse dataset of food and non-food images to ensure robust performance. It achieves high accuracy, making it suitable for applications in food recognition systems. This model can be easily integrated into various projects to automate the process of identifying food in images.

## Results

Through iterative model improvements and extensive testing, the model achieved a high classification accuracy on the test set. Detailed results and performance metrics can be found in the `results` directory


<img width="1469" alt="Screenshot 2024-07-30 at 9 48 46 PM" src="https://github.com/user-attachments/assets/0cca77ab-32bf-4288-b6c0-559b530eec27">





<img width="1470" alt="Screenshot 2024-07-30 at 9 52 49 PM" src="https://github.com/user-attachments/assets/9a8cddd5-2396-4d08-ae45-ab28f03f1b7c">





<img width="1470" alt="Screenshot 2024-07-30 at 9 53 44 PM" src="https://github.com/user-attachments/assets/852f9c8a-ab79-4c8d-b963-2bccd7a0b374">


## Contributing

Contributions are welcome! If you have any ideas, suggestions, or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The [Food 5K dataset](http://foodcam.mobi/dataset/) for providing the images.
- The contributors of various open-source libraries used in this project.

---
