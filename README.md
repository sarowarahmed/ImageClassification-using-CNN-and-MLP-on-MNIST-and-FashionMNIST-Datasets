# Image Classification using CNN and MLP on MNIST and Fashion MNIST Datasets

<h2 align="center">Overview</h2> 

*This project aims to implement and compare two deep learning models—Convolutional Neural Networks (CNN) and Multi-Layer Perceptrons (MLP)—for image classification on the MNIST and Fashion MNIST datasets. I experimented with different architectures, tuned hyperparameters, and analyzed model performance through visualizations and confusion matrices.*

Problem Statement: The goal of this project is to

1. Implement a CNN and MLP architecture for image classification.
2. Train the models on MNIST and Fashion MNIST datasets.
3. Explore different hyperparameters and configurations to improve model performance.
4. Compare the models in terms of accuracy, training time, and common misclassifications.

<h3 align="center">Steps Followed</h3>

<h2>A. Dataset Exploration</h2> 

*I loaded the MNIST and Fashion MNIST datasets using TensorFlow/Keras.
Visualized samples from both datasets to understand the image characteristics.*

<h2>B. CNN Model Implementation</h2> 

*We built a CNN architecture with two convolutional layers, max-pooling layers, and fully connected layers. The model was trained on both datasets.*

<h3>CNN Architecture:</h3>

1. Convolutional Layer with 32 filters (3x3) and ReLU activation.
2. MaxPooling Layer (2x2).
3. Convolutional Layer with 32 filters (3x3) and ReLU activation.
4. MaxPooling Layer (2x2).
5. Flatten Layer.
6. Dense Layer with 128 neurons and ReLU activation.
7. Output Layer with 10 neurons (softmax activation for classification).

<h3>Training Parameters:</h3>

1. Batch size: 64
2. Epochs: 10
3. Learning rate: 0.001

<h2>C. CNN Hyperparameter Tuning</h2> 

*To further improve model performance, we experimented with the following hyperparameters*

1. Increased filter sizes to 64.
2. Adjusted kernel size to (5x5) for better feature extraction.
3. Added dropout layers (0.5 rate) to reduce overfitting.
4. Lowered the learning rate to 0.0001 for more stable training.

*The tuned CNN model improved performance, particularly on the Fashion MNIST dataset.*

<h2>D. MLP Model Implementation</h2> 

*We implemented a Multi-Layer Perceptron (MLP) with the following architecture*

1. Flatten Layer to transform the image data.
2. Dense Layer with 128 neurons and ReLU activation.
3. Dropout Layer (0.5) for regularization.
4. Dense Layer with 64 neurons and ReLU activation.
5. Output Layer with 10 neurons (softmax activation for classification).

<h3>Training Parameters:</h3>

1. Batch size: 64
2. Epochs: 10
3. Learning rate: 0.001

<h2>E. MLP Hyperparameter Tuning</h2> 

*To improve the MLP's performance, we adjusted the following hyperparameters*

1. Increased the number of neurons in the fully connected layers (256, 128).
2. Reduced dropout rate to 0.3.
3. Lowered learning rate to 0.0001.

<h2>F. Model Comparison and Analysis</h2> 

1. We trained both models (CNN and MLP) on the MNIST and Fashion MNIST datasets.
2. Generated training/validation accuracy and loss curves to visualize model performance over epochs.
3. Generated confusion matrices to analyze common misclassifications in both models.

<h3>Results</h3>

1. CNN performed better on both datasets, achieving 99% accuracy on MNIST and 88% accuracy on Fashion MNIST.
2. MLP performed well but was slightly behind the CNN, with 97% accuracy on MNIST and 87% accuracy on Fashion MNIST.

<h2>G. Conclusion</h2> 

*The CNN model outperformed the MLP model, particularly on the Fashion MNIST dataset, which contains more complex image patterns. Hyperparameter tuning further improved both models' performance, and the analysis of confusion matrices highlighted areas of improvement, such as reducing misclassifications between similar classes.*

<h2 align="center">Project Structure</h3>


├── README.md

├── Image_Classification.ipynb

├── Problem Statement

├── Images

└── LICENSE


I. README.md: *Contains project overview and explanation.*

II. Image_Classification.ipynb: *Contains the  implementation and experiments CNN model & MLP model. Also Contains code for dataset loading and visualization.*

III. Problem Statement: *contains all the mere details on how to approach to the project*

IV. Images: *Folder containing visualization outputs (accuracy/loss curves, confusion matrices).*

V. LICENSE: *A short and simple permissive license with conditions only requiring preservation of copyright and license notices.* 

<h2 align="center">🌟 Exciting Update: Internship Completion! 🌟</h3>

This project was a part of the Internship (Machine Learning Intern at Skolar) which I have successfully completed during April 2024 to June 2024. 🎉

During this internship, I had the amazing opportunity to work on an Image Classification project using CNN and MLP on the MNIST and Fashion MNIST datasets. This project helped me enhance my skills in deep learning, hyperparameter tuning, and model analysis. The experience was both challenging and incredibly rewarding.

💡 Key Takeaways:

1. Mastered the application of CNNs and MLPs in image classification.
2. Gained a deeper understanding of hyperparameter tuning and model optimization.
3. Improved my ability to analyze and interpret model results using visualizations and confusion matrices.

📢 P.S. I have provided the intership certificate in the files.
