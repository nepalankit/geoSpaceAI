from flask import request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.models import load_model
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage import binary_dilation, binary_closing

# Load the flood segmentation model
def load_flood_model():
    try:
        # Try to load from backend directory first
        try:
            model = load_model('flood_segmentation_model.h5')
            print("Flood segmentation model loaded successfully from current directory")
            return model
        except:
            # Try to load from parent directory
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(parent_dir, 'flood_segmentation_model.h5')
            model = load_model(model_path)
            print(f"Flood segmentation model loaded successfully from {model_path}")
            return model
    except Exception as e:
        print(f"Error loading flood segmentation model: {str(e)}")
        return None

# Function to convert matplotlib figure to base64 string
def to_base64(img_array, cmap=None, title=None):
    plt.figure(figsize=(5, 5))
    if cmap:
        plt.imshow(img_array, cmap=cmap)
    else:
        plt.imshow(img_array)
    
    if title:
        plt.title(title)
    
    plt.axis('off')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')

# Function to generate visualization of segmentation results
def generate_segmentation_visualization(original_image, prediction):
    try:
        # Create a figure with three subplots side by side
        plt.figure(figsize=(15, 5))
        
        # Plot the original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Input Image')
        plt.axis('off')
        
        # Process the prediction mask
        if len(prediction.shape) == 4:  # Batch, height, width, channels
            mask = prediction[0, :, :, 0]  # Take first batch item, first channel
        else:
            mask = prediction.squeeze()
        
        # Apply Gaussian smoothing to the probability map for better visualization
        prob_mask_smooth = gaussian_filter(mask, sigma=0.5)
        
        # Plot the probability map
        plt.subplot(1, 3, 2)
        from matplotlib.colors import LinearSegmentedColormap
        prob_cmap = LinearSegmentedColormap.from_list('prob_cmap', ['#2D004B', '#FFFF00'], N=256)
        plt.imshow(prob_mask_smooth, cmap=prob_cmap, vmin=0, vmax=1)
        plt.title('Probability Map')
        plt.axis('off')
        
        # Add a colorbar to the probability map
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=prob_cmap), cax=cax)
        
        # Create binary mask for visualization
        thresh = 0.15  # Threshold for visualization
        binary_mask = (prob_mask_smooth > thresh).astype(np.uint8)
        
        # Apply morphological operations to clean up the mask
        binary_mask = binary_closing(binary_mask.astype(bool), structure=np.ones((3,3))).astype(bool)
        binary_mask = remove_small_objects(binary_mask, min_size=5)
        binary_mask = remove_small_holes(binary_mask, area_threshold=5)
        binary_mask = binary_dilation(binary_mask, iterations=3).astype(np.uint8)
        
        # Plot the binary mask
        plt.subplot(1, 3, 3)
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        # Save the visualization to a base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        visualization = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return visualization
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        return None

# Function to handle flood prediction requests
def predict_flood():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess the image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))  # Resize to match model input size
        img_array = np.array(img) / 255.0  # Normalize pixel values
        
        # Add batch dimension if needed
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Load the model
        model = load_flood_model()
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Process the prediction mask
        if len(prediction.shape) == 4:  # [batch, height, width, channels]
            mask = prediction[0, :, :, 0]  # Take first image, first channel
        else:
            mask = prediction.squeeze()
        
        # Use a threshold to create binary mask
        threshold = 0.25
        binary_mask = (mask > threshold).astype(np.uint8)
        
        # Apply morphological operations to clean up the mask
        binary_mask = binary_closing(binary_mask.astype(bool), structure=np.ones((3,3))).astype(bool)
        binary_mask = remove_small_objects(binary_mask, min_size=5)
        binary_mask = remove_small_holes(binary_mask, area_threshold=5)
        binary_mask = binary_dilation(binary_mask, iterations=3).astype(np.uint8)
        
        # Calculate confidence as mean probability of segmented class
        if np.sum(mask > 0.2) > 0:
            confidence = float(np.mean(mask[mask > 0.2]))
        else:
            confidence = float(np.mean(mask))
        
        # Determine result based on mask content and confidence
        if np.sum(binary_mask) > 50 and confidence > 0.3:
            result = 'flood area detected'
        else:
            if np.sum(mask > 0.25) > 20:
                result = 'potential flood area detected (low confidence)'
                confidence = max(confidence, 0.3)
            else:
                result = 'no flood area detected'
                confidence = min(confidence, 0.2)
        
        # Generate visualization
        segmentation_image = generate_segmentation_visualization(img_array[0], prediction)
        
        # Create result images for display
        result_images = {
            'rgb': to_base64(img_array[0]),
            'mask': to_base64(binary_mask, cmap='gray', title="Binary Mask"),
            'prediction': to_base64(mask, cmap='viridis', title="Prediction")
        }
        
        return jsonify({
            'result': result,
            'confidence': confidence,
            'segmentation_image': segmentation_image,
            'result_images': result_images
        })
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error during flood prediction: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({'error': str(e), 'details': error_traceback}), 500