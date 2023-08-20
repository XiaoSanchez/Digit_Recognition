# Streamlit Digit Recognition

This repository contains a web-based digit recognition application built using the Streamlit library. The app allows users to draw a digit on a canvas and predicts the digit's class using a trained convolutional neural network (CNN) model. The model has been trained on the MNIST dataset for digit classification.

## Installation

To run the digit recognition web app locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/digit-recognition-app.git
   cd digit-recognition-app
   ```

2. Set up a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Once the app is running, you'll see the title "Digit Recognizer" displayed.

2. Use the drawing canvas to draw a digit using your mouse or touch input. You can customize the fill and stroke colors, background color, canvas size, and drawing mode.

3. After drawing a digit, click the "Predict" button to initiate the digit recognition process.

4. The app will process the drawn image, resize it to 28x28 pixels, and convert it to grayscale. It will then pass the image through the trained CNN model to obtain raw predictions.

5. The predicted probabilities for each digit class (0-9) will be displayed in a bar chart. The x-axis labels correspond to the digit classes, and the y-axis shows the probabilities scaled by 100.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to customize this template further based on your specific project needs. It provides users with clear instructions for installation, usage, contribution, and license information. Make sure to replace placeholders like `your-username` with your actual GitHub username and update any filenames or paths as necessary to match your project's structure. Good luck with your open-source project!