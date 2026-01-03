# Plant Disease Detection Web Application

A web-based plant disease detection system using deep learning. This application helps identify diseases in plant leaves using an EfficientNet model trained on the PlantVillage dataset.

## Features

- **Web-based Interface**: Easy-to-use web interface for uploading and analyzing plant leaf images
- **High Accuracy**: Powered by an EfficientNet-B0 model trained on thousands of images
- **Detailed Analysis**: Provides disease identification, confidence scores, and management recommendations
- **Plant Information**: Includes a database of information about common plant diseases
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Supported Plants and Diseases

The system can identify diseases in several plant species including:
- Tomato
- Potato
- Apple
- Corn
- Grape
- Cherry
- Pepper
- Strawberry
- and more...

Common detectable diseases include:
- Late Blight
- Early Blight
- Black Rot
- Powdery Mildew
- Leaf Scorch
- Common Rust
- and many others...

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Flask 2.2+
- NumPy, Pillow, scikit-learn
- Other dependencies listed in `requirements.txt`

### Setup

1. Extract the source code zip
   
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the PlantVillage dataset:
   - Visit [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
   - Download and extract to `dataset/PlantVillage`
   
   Alternatively, if you have Kaggle API configured:
   ```
   kaggle datasets download -d abdallahalidev/plantvillage-dataset
   unzip plantvillage-dataset.zip -d dataset/
   ```

## Usage

### Training the Model

To train the model, run:
```
python train_model.py
```

If you want to force retraining even if a trained model exists:
```
python train_model.py --force
```

### Running the Web Application

To launch the web application:
```
python app.py
```


Then access the application in your browser at:
```
http://localhost:5001
```

### Using the Application

1. Navigate to the "Detect Disease" page
2. Upload an image of a plant leaf
3. The system will automatically analyze the image
4. View the results and recommendations

## Project Structure

```
plant-disease-detection/
├── app.py                   # Flask application
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── best_model.pth           # Trained model (generated after training)
├── class_indices.npy        # Class mapping file (generated after training)
├── model_metadata.json      # Model information (generated after training)
├── static/                  # Static assets
│   ├── css/                 # Stylesheets
│   ├── js/                  # JavaScript files
│   └── images/              # Application images and icons
├── templates/               # HTML templates
│   ├── base.html            # Base template
│   ├── index.html           # Home page
│   ├── detect.html          # Disease detection page
│   ├── diseases.html        # Plant diseases information page
│   ├── about.html           # About page
└── dataset/                 # Dataset folder
    └── PlantVillage/        # Plant disease images
```

## Development

### Extending the Model

To add support for new plant diseases:
1. Add labeled examples to the training dataset
2. Retrain the model using `python train_model.py --force`

### Improving the Application

The web application is built with Flask and can be extended by modifying the appropriate templates and routes.


## Acknowledgments

- The PlantVillage dataset by Penn State University
- EfficientNet model architecture by Google Research
- PyTorch framework by Facebook AI Research
- Flask web framework

