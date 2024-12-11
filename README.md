# Machine Learning Project: Food Image Classification with 19 Classes

## Project Structure:

<img width="902" alt="Screenshot 2024-12-11 at 15 32 24" src="https://github.com/user-attachments/assets/a57f5dd8-862e-49ad-b50d-db8c2d6f49e9">

## Description :
This project aims to build and train an image classification model using **Convolutional Neural Networks (CNN)**. The model classifies images into 19 different food categories.
#### 19 Classes:
`apple`, `avocado`, `banana`, `broccoli`, `carrot`, `chicken`, `corn`, `dragon fruit`, `egg`, `grape`, `green vegetables`, `orange`, `porridge`,     `potato`, `rice`, `tempeh`, `tofu`, `tomato`, `watermelon`

- <img width="333" alt="Screenshot 2024-12-11 at 15 01 50" src="https://github.com/user-attachments/assets/2f9c648a-798d-400d-ab38-1caf22c148b0">

# Tools Used:
### TensorFlow, Keras for building and training the CNN model.
### OpenCV for image processing.

# Data Preprocessing
* Image augmentation to increase data variety and model performance.
* Each class contains approximately 1,000 images, totaling 19,000 images.
* After augmentation, the dataset is divided into:
    * Training: 15,200 images
    * Validation: 1,900 images
    * Test: 1,900 images

##### Augmentasi Gambar
     - Images are processed through augmentation to increase training data variability and improve model performance using `ImageDataGenerator` from Keras with the following parameters:
         - Rescale: All pixel values are normalized by dividing by 255, so the pixel values range from 0 to 1.
         - Rotation Range: Images can be randomly rotated within a range of 20 degrees.
         - Width Shift Range and Height Shift Range: Images can be randomly shifted horizontally and vertically within a range of 0.2.
         - Shear Range: Applies random shear transformations to the images within a range of 0.2.
         - Zoom Range: Images can be zoomed in or out within a range of 0.2.
         - Horizontal Flip: Images can be randomly flipped horizontally.


# Full Notebook Version
This is the full version of the project in .ipynb format, which can be used for exploration from start to output, including model building, training, and evaluation in more detail. All processes used in model training, evaluation, and visualization are included in the **notebook.ipynb** file.

# Script .py
**Note:** Provides a separate .py file focused on testing the model using images outside of the dataset: 
   1. **predict_model.py** (to test the model that was generated during training using 50 epochs, which results in the **best_model.keras** file)
   

# Model Architecture:
The model uses **CNN** for image classification with 
            - **4 convolution layers** (Conv2D) to extract features, followed by batch normalization and max pooling to reduce dimensionality.                - **GlobalAveragePooling2D** is used for further dimensionality reduction before entering the Dense layers for final classification.              - **Dropout** is applied to prevent overfitting.
            
## Model Compilation:
The model is compiled with the Adam optimizer.

# Model Evaluation
Model performance is measured using a Confusion Matrix that provides insight into how well the model classifies images into the correct classes. Below is the confusion matrix generated after the model was tested on the test dataset:

   <img width="869" alt="5_confusion_matrix" src="https://github.com/user-attachments/assets/51fca590-b652-4aab-baaf-ac8a24ab3c7a">

## Saving the Best Model:
The best model is automatically saved during training as **best_model.keras**. This model file can then be converted to .h5 format for use in production or testing environments by MD and CC team.

# Requirements File:
The requirements file to run this project can be found in a file named **requirements.txt**. You can install all the necessary dependencies with this file.


# How to Clone the GitHub Project:
##### Ensure Git is installed on your computer. If not, install Git first.
## 1. Clone the Repository
Open the Terminal in VSCode, and run the following command to clone the repository:

            - git clone https://github.com/Kiddos-Kids-Food-Discoveries/kiddos-ml.git

## 2. Enter the Project Folder
Once the cloning process is complete, enter the project folder with the following command:

            - cd kiddos-ml
## 3. Check the Folder Contents
To verify the folder and files inside, run the command:

            - ls
## 4. Create and Activate a Virtual Environment (Optional, but recommended):
To install all the dependencies required for this project, use pip by following these steps:
###### For macOS/Linux:
            - python -m venv venv
            - source venv/bin/activate
###### For Windows:
            - python -m venv venv
            - .\venv\Scripts\activate

## 5. Install Dependencies:
Once the virtual environment is active, run the following command to install all dependencies listed in **requirements.txt**:

            - pip install -r requirements.txt

## 6. Run the Project:
### - Running the Jupyter Notebook (notebook.ipynb):
To run the Jupyter notebook, execute the following command in the terminal:

            - jupyter notebook

### - Running the Python Script (predict_model.py):
If you want to run the Python file like **predict_model.py** to directly test the model with images outside the prepared dataset, run the following command in the terminal:

            - python predict_model.py




# Good luck! 😉
