# geoSpaceAI

## Project Overview

This repository contains code and resources for a geoSpatial AI project focusing on landslide and flood detection using deep learning and related technologies. The project leverages JavaScript, Python, and TypeScript, along with frameworks and tools like Node.js, to develop functionalities for image processing, model deployment, and user interface creation.

## Key Features & Benefits

- **Landslide Detection:** Utilizes machine learning models (implemented in Jupyter notebooks) to identify potential landslide areas.
- **Flood Prediction & Segmentation:** Employs Python and deep learning models to predict and segment flood-prone areas from images.
- **Interactive User Interface:** A React-based front-end provides users with tools to upload images, visualize predictions on a map, and access model information.
- **Backend API:** Flask-based API serves the machine learning models and provides endpoints for image processing and prediction.
- **Data Visualization:** Integrated map views and data visualization components offer insights into the analyzed geospatial data.

## Prerequisites & Dependencies

Before setting up the project, ensure you have the following installed:

- **Node.js:** JavaScript runtime environment.
- **npm** or **bun:** Package managers for installing JavaScript dependencies.
- **Python 3.6+:** Python programming language.
- **pip:** Python package installer.
- **Git:** Version control system.

The project relies on the following key dependencies:

**Frontend (React):**
- React
- TypeScript
- Tailwind CSS
- ESLint

**Backend (Python):**
- Flask
- TensorFlow/Keras
- Pillow (PIL)
- Flask-CORS
- h5py
- scikit-image
- matplotlib

## Installation & Setup Instructions

Follow these steps to install and set up the project:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nepalankit/geoSpaceAI.git
   cd geoSpaceAI
   ```

2. **Backend Setup (Python):**

   - Navigate to the `backend` directory:

     ```bash
     cd backend
     ```

   - Create a virtual environment (optional but recommended):

     ```bash
     python3 -m venv venv
     source venv/bin/activate  # On Linux/macOS
     venv\Scripts\activate  # On Windows
     ```

   - Install Python dependencies:

     ```bash
     pip install -r requirements.txt
     ```

   - Start the Flask server:

     ```bash
     python app.py
     ```

3. **Frontend Setup (React):**

   - Navigate to the project's root directory:

     ```bash
     cd .. # Go back to the root directory
     ```

   - Install Node.js dependencies:

     ```bash
     npm install # or bun install
     ```

   - Start the development server:

     ```bash
     npm run dev # or bun dev
     ```

   - The React app should now be running on a local development server (e.g., `http://localhost:5173`).

## Usage Examples & API Documentation

### Frontend (React):
The frontend provides a user interface for:

- **Uploading images:** The `ImageUpload.tsx` component allows users to upload images for analysis.
- **Visualizing results:** The `MapView.tsx` component displays the processed images and predictions on a map interface.
- **Accessing model information:** The `ModelInfo.tsx` component provides details about the machine learning models used.

### Backend (Flask API):

The Flask API provides the following endpoints:

- **`/predict_landslide`**: Accepts an image and returns landslide predictions. (Details in `backend/app.py` and `backend/flood_prediction.py`)
- **`/predict_flood`**: Accepts an image and returns flood predictions. (Details in `backend/app.py` and `backend/flood_prediction.py`)

Example using `curl`:
```bash
curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:5000/predict_flood
```

## Configuration Options

### Backend (Flask):

- The Flask server can be configured by modifying the `app.py` file.
- You can set environment variables for API keys or other sensitive information.
- The model paths in `app.py` should be updated to reflect the location of your trained models.

### Frontend (React):

- The React app can be configured by modifying the environment variables in the `.env` file.
- You can customize the map settings in the `MapView.tsx` component.

## Contributing Guidelines

We welcome contributions to the project! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Submit a pull request to the main branch.

Please follow the existing code style and conventions.

## License Information

All rights reserved.

## Acknowledgments

- This project utilizes various open-source libraries and tools, including React, Flask, TensorFlow, and others.
- We thank the developers and maintainers of these libraries for their contributions to the open-source community.
