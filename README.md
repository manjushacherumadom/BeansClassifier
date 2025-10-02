# BeansClassifier
To help farmers identify diseased beans easily, this Streamlit application can help where users can upload an image of a bean and receive a prediction of its class
The class names indicate a classification problem with 3 classes.
The images are color images (3 channels)
angular_leaf_spot
bean_rust
healthy
<img width="940" height="193" alt="image" src="https://github.com/user-attachments/assets/059030fe-728a-4245-b996-a16522591ab8" />
To apply transfer learning with MobileNet on the Beans dataset, we'll use the pre-trained MobileNet model, which has been trained on ImageNet. 
The goal is to leverage the learned features (such as textures, edges, shapes, etc.) from ImageNet and adapt them to our bean diseases identification task.

TRAINING THE MODEL

When training a machine learning model, overfitting occurs when the model starts to memorize the training data rather than learning generalizable patterns. To prevent overfitting, we can use Early Stopping, a regularization technique that monitors the model's performance on a validation set during training. If the model's performance on the validation set does not improve for a certain number of epochs, the training process is stopped early. This helps in preventing the model from overfitting and saves computational resources by halting the training process once the model reaches its best performance

MODEL EVALUATION

To evaluate the trained model on the test dataset and assess its performance in classifying new bean images, we will calculate the key metrics such as accuracy, precision, and recall.

MODEL SAVING AND REUSABILITY

model.save('my_bean_disease_classifier.keras')

This saves the model's architecture, optimizer state, and learned weights to a file named my_bean_disease_classifier.keras. It’s important to note that this saved model can now be reused and loaded in the future without needing to retrain it.
The saved .keras file contains all the information necessary to restore the model, including the model architecture (layers and how they connect),the learned weights
,the optimizer settings (if any),the training configuration (for future fine-tuning).

Steps to Load and Reuse the Model

Once the model has been saved, it can easily be loaded and reused for prediction, evaluation, or further fine-tuning. Here’s how you can load and reuse the model:
	1. Load the Model in a New Environment
You can load the saved model in a different environment or by other team members by  using the tensorflow.keras.models.load_model() function. This will restore both the architecture and the weights of the model.
	2. Use the Loaded Model for Prediction or Evaluation
 Once the model is loaded, you can use it for predictions or evaluate it on new data, just   as you would with the original model
	3. Fine-Tune the Loaded Model 
If you want to continue training or fine-tuning the loaded model, you can do so by compiling the model again and continuing the training process.

Why Use the .keras Format?

•	Portability: The .keras format is native to TensorFlow and can be easily shared across different environments and machines, ensuring compatibility.
•	Efficiency: It saves both the model’s architecture and weights in a compact and efficient way, making it suitable for deployment and future use.
•	Flexibility: The .keras format can be loaded and used directly for inference or further training.

Explanation of the Steps:

1.	Saving the Model:
Why Save the Model? Saving the model after training allows you to preserve the learned weights, architecture, and configurations. This ensures that you don’t have to retrain the model from scratch each time you want to deploy it or make predictions, saving both time and resources.
The .keras format is a self-contained format that stores everything in a single file, making it ideal for sharing, storing, and deploying.
2.	Loading the Model:
After saving, you can load the model in any environment with TensorFlow installed. This is helpful for transferring the model to production systems, collaborating with other team members, or performing inference on new datasets.
The load_model() function will automatically reconstruct the model architecture and restore the learned weights, so you don't need to rebuild the model manually.
3.	Use in Different Environments:
When loading the model in a different environment, it’s essential to ensure that the TensorFlow version is compatible. For example, if your team uses different versions of TensorFlow, make sure the version is consistent or the model was saved in a way that ensures backward compatibility.
4.	File Management: You might want to save the model in a directory or cloud storage (like   AWS S3, GCP Cloud Storage, or Google Drive) to allow access from different environments or team members.
5.	Model Versioning: When deploying multiple models or versions, it's a good idea to version your saved models (e.g., model_v1.keras, model_v2.keras) to keep track of changes and improvements.
6.	Dependencies: If you plan to load the model in a different environment, make sure that the necessary libraries (e.g., TensorFlow, Keras) and dependencies are installed
Summary
•	Saving the model: Use model.save('model_name.keras') to save the trained model in the .keras format.
•	Loading the model: Use load_model('model_name.keras') to load the saved model into a new environment.
•	Reuse: Once loaded, the model can be used for predictions, evaluation, or further fine-tuning.
This process ensures that your model can be easily transferred and reused, making it suitable for deployment in production systems or sharing with team members for collaboration.

DEPLOYING A STREAMLIT APPLICATION

Creating a Streamlit application that allows farmers to upload images of beans and get predictions on the presence of diseases can be done using the trained model.
We'll create a simple Streamlit application where the user can upload an image, and the model will classify it into one of the three classes: angular_leaf_spot, bean_rust, or healthy.

How the Streamlit Application Works:

1.	Upload Image: The user uploads an image file via the st.file_uploader() widget. The file type is restricted to .jpg, .png, and .jpeg.
2.	Preprocess the Image: The uploaded image is resized to 224x224 pixels because MobileNet requires this input size. The image is converted into a NumPy array, normalized to a range of [0, 1], and reshaped to add a batch dimension, which is required for model input.
3.	Make Prediction: The pre-processed image is passed to the trained model for prediction. The model outputs a set of probabilities for each class, and we use np.argmax() to find the class with the highest probability. The predicted class and its corresponding confidence score (percentage) are displayed.
4.	Display Predictions: The application shows the predicted class label and confidence score. Optionally, it also displays the probabilities for all the classes to give more detailed information about the model’s confidence.

Key Components

file_uploader: Allows users to upload an image.

Image.open: Used to open the uploaded image file.

predict(): Uses the trained model to make predictions based on the uploaded image.

To Run the Application:

1.	Save the above code into a Python script file, for example, bean_disease_classifier.py.
2.	In the same directory, make sure you have the saved model file (my_bean_disease_classifier.keras).
3.	In the terminal, navigate to the directory where the script is saved and run:
4.	bash
5.	Copy code
6.	streamlit run bean_disease_classifier.py
   
This will launch the Streamlit app in your browser, allowing you to upload an image and get predictions.
This Streamlit application makes it easy for farmers or agricultural experts to use machine learning to diagnose bean leaf diseases based on image input.
