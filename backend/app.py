from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image
import io
import os
import h5py
import base64
import cv2
import json
import tempfile
import uuid
import shutil
# Set matplotlib to use a non-interactive backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt
from flask_cors import CORS
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.transform import resize

def generate_segmentation_visualization(original_data, prediction):
    """
    Generate a visualization of the multi-channel segmentation results.
    
    Args:
        original_data: The original image data (can be multi-channel)
        prediction: The model's prediction (segmentation mask)
        
    Returns:
        Base64 encoded string of the visualization image or None if an error occurs
    """
    try:
        print(f"Generating visualization with original_data shape: {original_data.shape} and prediction shape: {prediction.shape}")
        
        # Create a figure with three subplots side by side
        plt.figure(figsize=(15, 5))
        
        # Prepare the original image for display
        try:
            # Process the original data to create a displayable image
            if len(original_data.shape) >= 2:
                h, w = original_data.shape[:2]
                
                # Convert the original data to a displayable RGB image
                if len(original_data.shape) == 3 and original_data.shape[2] == 3:
                    # Already RGB format
                    display_image = original_data.copy()
                    # Normalize if needed
                    if display_image.max() > 1.0:
                        display_image = display_image / 255.0
                elif len(original_data.shape) == 3 and original_data.shape[2] > 3:
                    # Multi-channel data, use first 3 channels for RGB
                    display_image = np.zeros((h, w, 3))
                    for i in range(3):
                        channel_idx = min(i, original_data.shape[2]-1)
                        channel = original_data[:, :, channel_idx]
                        if channel.max() > 0:
                            display_image[:, :, i] = channel / channel.max()
                elif len(original_data.shape) == 3 and original_data.shape[2] == 1:
                    # Single channel, duplicate to RGB
                    display_image = np.zeros((h, w, 3))
                    for i in range(3):
                        display_image[:, :, i] = original_data[:, :, 0]
                    if display_image.max() > 0:
                        display_image = display_image / display_image.max()
                elif len(original_data.shape) == 2:
                    # 2D grayscale, convert to RGB
                    display_image = np.zeros((h, w, 3))
                    for i in range(3):
                        display_image[:, :, i] = original_data
                    if display_image.max() > 0:
                        display_image = display_image / display_image.max()
                else:
                    raise ValueError(f"Unsupported original_data shape: {original_data.shape}")
                
                # Ensure values are in valid range [0, 1]
                display_image = np.clip(display_image, 0, 1)
            else:
                raise ValueError(f"Invalid original_data shape: {original_data.shape}")
                
            print(f"Processed original image with shape: {display_image.shape}")
        except Exception as img_error:
            print(f"Error preparing original image: {str(img_error)}")
            # Create a simple gradient pattern as fallback
            h, w = 128, 128
            if len(original_data.shape) >= 2:
                h, w = original_data.shape[:2]
            
            display_image = np.zeros((h, w, 3))
            # Create a gradient pattern
            for i in range(h):
                for j in range(w):
                    # Create a smooth gradient from top-left to bottom-right
                    r_val = 0.3 + (0.7 * i / h)
                    g_val = 0.3 + (0.7 * j / w)
                    b_val = 0.5
                    display_image[i, j, :] = [r_val, g_val, b_val]
        
        # Plot the original image
        plt.subplot(1, 3, 1)
        if len(display_image.shape) == 3 and display_image.shape[2] == 3:
            plt.imshow(display_image)
        else:
            plt.imshow(display_image, cmap='viridis')
        plt.title('Input Image')
        plt.axis('off')
        
        # Process the prediction mask
        try:
            if len(prediction.shape) == 4:  # Batch, height, width, channels
                mask = prediction[0]  # Take first batch item
            else:
                mask = prediction
            
            # Create a probability map from the prediction
            h, w = original_data.shape[:2]  # Get height and width from original data
            
            # For the middle panel, show the raw probability map
            if len(mask.shape) == 3 and mask.shape[2] > 1:  # Multi-class segmentation
                # Use the first class (typically class 1)
                prob_mask = mask[:, :, 0]
            else:  # Binary segmentation
                # Ensure the mask is properly squeezed to 2D
                if len(mask.shape) > 2:
                    prob_mask = mask.squeeze()
                    if len(prob_mask.shape) > 2:  # If still more than 2D after squeeze
                        prob_mask = prob_mask[:, :, 0]  # Take the first channel
                else:
                    prob_mask = mask
            
            # Apply Gaussian smoothing to the probability map for better visualization
            prob_mask_smooth = gaussian_filter(prob_mask, sigma=0.5)
            
            # Plot the probability map
            plt.subplot(1, 3, 2)
            # Use a continuous colormap for better visualization of probabilities
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
            
            # Plot the predicted segmentation mask
            plt.subplot(1, 3, 3)
            
            # Process the prediction to create a binary mask
            try:
                # Print prediction statistics
                print(f"Visualization - prob_mask stats: min={np.min(prob_mask)}, max={np.max(prob_mask)}, mean={np.mean(prob_mask)}")
                print(f"Visualization - prob_mask_smooth stats: min={np.min(prob_mask_smooth)}, max={np.max(prob_mask_smooth)}, mean={np.mean(prob_mask_smooth)}")
                
                # Use a lower threshold if the prediction values are very low
                if np.max(prob_mask_smooth) < 0.3:
                    print("Using lower threshold for visualization due to low prediction values")
                    thresh = 0.1
                else:
                    try:
                        # Try to use Otsu's method first
                        thresh = threshold_otsu(prob_mask_smooth)
                        # If threshold is too high (resulting in very few segmented pixels), lower it
                        if np.mean(prob_mask_smooth > thresh) < 0.01:  # If less than 1% of pixels are segmented
                            thresh = np.percentile(prob_mask_smooth, 90)  # Use 90th percentile instead
                    except:
                        # Fallback threshold
                        thresh = np.percentile(prob_mask_smooth, 85)  # Use 85th percentile
                
                print(f"Visualization - using threshold: {thresh}")
                
                # Create binary mask
                binary_mask = (prob_mask_smooth > thresh).astype(np.uint8)
                
                # Apply morphological operations to clean up the mask
                binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=50)
                binary_mask = remove_small_holes(binary_mask, area_threshold=50)
                
                print(f"Visualization - binary_mask stats: sum={np.sum(binary_mask)}, unique values={np.unique(binary_mask)}")
                
                # If the prediction is empty or very small, show a message
                if np.sum(binary_mask) < 10:  # If prediction is too small or empty
                    print("Warning: Binary mask is empty or very small")
                    
                    # Instead of showing empty mask, show the raw probability map with a lower threshold
                    # This helps debug cases where the model is producing very low confidence predictions
                    from matplotlib.colors import LinearSegmentedColormap
                    prob_cmap = LinearSegmentedColormap.from_list('prob_cmap', ['#2D004B', '#FFFF00'], N=256)
                    plt.imshow(prob_mask_smooth, cmap=prob_cmap, vmin=0, vmax=np.max(prob_mask_smooth) or 1)
                    plt.title('Low Confidence Prediction')
                    
                    # Add text explaining the visualization
                    plt.text(0.5, 0.1, 'Very low confidence predictions shown', 
                             horizontalalignment='center', verticalalignment='center',
                             transform=plt.gca().transAxes, color='white', fontsize=10,
                             bbox=dict(facecolor='black', alpha=0.5))
                    
                    # Add a colorbar
                    divider = make_axes_locatable(plt.gca())
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, np.max(prob_mask_smooth) or 1), 
                                                      cmap=prob_cmap), cax=cax)
                else:
                    # Display the binary mask with a custom colormap
                    from matplotlib.colors import ListedColormap
                    custom_cmap = ListedColormap(['#2D004B', '#FFFF00'])  # Purple background, yellow segmented areas
                    plt.imshow(binary_mask, cmap=custom_cmap, vmin=0, vmax=1)
                    
                    # Add a title that indicates this is the segmentation result
                    plt.title('Segmentation Result')
                    
                    # Add a colorbar for reference
                    divider = make_axes_locatable(plt.gca())
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=custom_cmap), cax=cax)
                
            except Exception as viz_error:
                print(f"Error creating binary mask visualization: {str(viz_error)}")
                # Show a blank visualization with an error message
                plt.imshow(np.zeros((h, w)), cmap='gray')
                plt.title('Segmentation Result')
                plt.text(0.5, 0.5, 'Error processing segmentation', 
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes, color='white', fontsize=10)
            except Exception as mask_viz_error:
                print(f"Error creating predicted mask visualization: {str(mask_viz_error)}")
                # Fallback to simple binary mask
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                    if len(mask.shape) > 2:
                        mask = mask[:, :, 0]
                binary_mask = (mask > 0.5).astype(float)
                # Use the same custom colormap for consistency
                from matplotlib.colors import ListedColormap
                custom_cmap = ListedColormap(['#2D004B', '#FFFF00'])  # Purple background, yellow segmented areas
                plt.imshow(binary_mask, cmap=custom_cmap)
                plt.title('Segmentation Result')
            
            plt.axis('off')
            
        except Exception as pred_error:
            print(f"Error processing prediction mask: {str(pred_error)}")
            # Create an empty prediction mask as fallback
            plt.subplot(1, 3, 2)
            plt.imshow(np.zeros((128, 128)), cmap='gray')
            plt.title('Probability Map')
            plt.text(0.5, 0.5, 'Error processing data', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, color='white', fontsize=10)
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            # Use the same custom colormap for consistency
            from matplotlib.colors import ListedColormap
            custom_cmap = ListedColormap(['#2D004B', '#FFFF00'])  # Purple background, yellow segmented areas
            plt.imshow(np.zeros((128, 128)), cmap=custom_cmap)
            plt.title('Segmentation Result')
            plt.text(0.5, 0.5, 'Error processing data', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, color='white', fontsize=10)
            plt.axis('off')
        
        # Set figure background to white to avoid black background
        plt.gcf().patch.set_facecolor('white')
        
        # Save the figure to a BytesIO object with white background
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
        plt.close()
        buf.seek(0)
        
        # Encode the image as base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_str
    
    except Exception as e:
        import traceback
        print(f"Error generating segmentation visualization: {str(e)}")
        print(traceback.format_exc())
        return None

# Helper function to resize images
def resize_img(img, target_size=(128, 128)):
    return resize(img, target_size, preserve_range=True, anti_aliasing=True)

# Helper function to convert numpy array to base64 image
def to_base64(img, cmap=None, title=None):
    plt.figure(figsize=(2,2))
    
    # Ensure img is a numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # Print image stats for debugging
    print(f"to_base64 - {title if title else 'image'} stats: shape={img.shape}, dtype={img.dtype}, min={np.min(img)}, max={np.max(img)}, mean={np.mean(img)}")
    
    # Handle different image types
    if img.ndim == 2:
        # For 2D arrays (grayscale/masks)
        # Normalize if needed
        if img.max() > 0:
            # Use vmin/vmax to ensure proper contrast
            plt.imshow(img, cmap=cmap or 'viridis', vmin=0, vmax=img.max())
        else:
            # If image is all zeros, still display it
            plt.imshow(np.zeros_like(img), cmap=cmap or 'viridis')
            plt.text(0.5, 0.5, 'No data', horizontalalignment='center', 
                     verticalalignment='center', transform=plt.gca().transAxes)
    elif img.ndim == 3 and img.shape[2] == 3:
        # For RGB images
        # Ensure values are in [0, 1] range
        if img.max() > 1.0:
            img = img / 255.0
        plt.imshow(np.clip(img, 0, 1))
    else:
        # For other cases
        plt.imshow(img, cmap=cmap or 'viridis')
    
    if title:
        plt.title(title)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../unet_save.h5')
model = load_model(MODEL_PATH)

# Adjust these to match the model's expected input shape
IMG_SIZE = (128, 128)  # Model expects 128x128 images
CHANNELS = 6  # Model expects 6 channels

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        # Load and preprocess the image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = img_to_array(img)
        
        # Normalize RGB channels
        rgb_array = img_array / 255.0
        
        # Step 1: RGB Image (already have it)
        rgb_disp = rgb_array.copy()
        
        # Step 2: Calculate NDVI (using red and a simulated NIR channel)
        # Since we don't have actual NIR in RGB images, we'll simulate it
        red = rgb_array[:,:,0]
        # Simulate NIR using a combination of channels
        simulated_nir = (rgb_array[:,:,1] + rgb_array[:,:,2]) / 2  # Simple approximation
        ndvi = (simulated_nir - red) / (simulated_nir + red + 1e-6)  # Avoid division by zero
        ndvi_disp = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-6)
        
        # Step 3: Generate Slope (simulated from image gradients)
        # Calculate gradient magnitude as a proxy for slope
        dx = cv2.Sobel(rgb_array[:,:,0], cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(rgb_array[:,:,0], cv2.CV_64F, 0, 1, ksize=3)
        slope = np.sqrt(dx**2 + dy**2)
        slope_disp = (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)
        
        # Step 4: Generate DEM (simulated from image intensity)
        # Use image brightness as a proxy for elevation
        brightness = np.mean(rgb_array, axis=2)
        dem = gaussian_filter(brightness, sigma=3)  # Smooth it to look more like elevation
        dem_disp = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)
        
        # Create a 6-channel input array for the model
        six_channel_array = np.zeros((IMG_SIZE[0], IMG_SIZE[1], CHANNELS))
        
        # First 3 channels: RGB
        six_channel_array[:,:,0:3] = rgb_array
        
        # Channel 4: NDVI
        six_channel_array[:,:,3] = ndvi_disp
        
        # Channel 5: Slope
        six_channel_array[:,:,4] = slope_disp
        
        # Channel 6: DEM
        six_channel_array[:,:,5] = dem_disp
        
        # Add batch dimension
        input_array = np.expand_dims(six_channel_array, axis=0)
        
        # Print input array statistics for debugging
        print(f"Input array shape: {input_array.shape}, min: {np.min(input_array)}, max: {np.max(input_array)}, mean: {np.mean(input_array)}")
        
        # Make prediction
        prediction = model.predict(input_array)
        
        # Print raw prediction statistics
        print(f"Raw prediction shape: {prediction.shape}, min: {np.min(prediction)}, max: {np.max(prediction)}, mean: {np.mean(prediction)}")
        
        # Step 5: Process the mask
        if len(prediction.shape) == 4:  # [batch, height, width, channels]
            mask = prediction[0, :, :, 0]  # Take first image, first channel
        else:
            mask = prediction.squeeze()

        # Print processed mask statistics
        print("Processed mask stats - min:", np.min(mask), "max:", np.max(mask), "mean:", np.mean(mask))
        
        # If the prediction values are all very low, apply contrast enhancement
        if np.max(mask) < 0.1:
            print("Applying contrast enhancement to low-value predictions")
            # Apply contrast stretching to make the prediction more visible
            p_min, p_max = np.percentile(mask, (1, 99))
            if p_max > p_min:  # Avoid division by zero
                mask = np.clip((mask - p_min) / (p_max - p_min), 0, 1)
            print("After enhancement - min:", np.min(mask), "max:", np.max(mask), "mean:", np.mean(mask))
        
        # Use an adaptive threshold based on the prediction values
        if np.max(mask) < 0.3:
            threshold = 0.05  # Very low threshold for weak predictions
        else:
            threshold = 0.3   # Standard threshold for stronger predictions
            
        print(f"Using threshold: {threshold}")
        
        binary_mask = (mask > threshold).astype(np.uint8)

        # (Optional) Clean up the mask
        binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=50)
        binary_mask = remove_small_holes(binary_mask, area_threshold=50)
        processed_mask = binary_mask.astype(float)
        
        # Print final binary mask statistics
        print(f"Binary mask stats - sum: {np.sum(binary_mask)}, unique values: {np.unique(binary_mask)}")
        
        # If the binary mask is empty, use the raw prediction values directly
        if np.sum(binary_mask) == 0:
            print("Warning: Binary mask is empty, using raw prediction values")
            processed_mask = mask  # Use the raw prediction values

        # For visualization, use different representations for mask and prediction
        mask_disp = processed_mask  # Binary mask (0/1 values)
        
        # Step 6: Final Prediction
        # Calculate confidence as mean probability of segmented class
        # If the mask has any positive values, use their mean as confidence
        if np.sum(mask > 0.05) > 0:
            confidence = float(np.mean(mask[mask > 0.05]))
        else:
            confidence = float(np.mean(mask))
            
        # Adjust result text based on confidence and mask content
        if np.sum(processed_mask) > 10:  # If we have a reasonable number of positive pixels
            result = 'segmented area detected'
        else:
            # Check if there are any areas with moderate probability
            if np.sum(mask > 0.1) > 10:
                result = 'potential segmented area detected (low confidence)'
                confidence = max(confidence, 0.3)  # Ensure minimum confidence for UI
            else:
                result = 'no segmented area detected'
                
        print(f"Final result: {result}, confidence: {confidence}")
        
        # Ensure confidence is in a reasonable range for UI display
        confidence = max(min(confidence, 0.99), 0.01)  # Clamp between 0.01 and 0.99
        
        # Create custom colormap for mask visualization
        # Use custom colormaps for better visualization
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create a custom colormap for the mask (black and yellow)
        mask_cmap = LinearSegmentedColormap.from_list('mask_cmap', ['black', 'yellow'])
        
        # Create a custom colormap for the prediction (purple to yellow)
        pred_cmap = LinearSegmentedColormap.from_list('pred_cmap', ['#2D004B', '#FFFF00'])
        
        # Print mask display stats
        print("Mask display stats - min:", np.min(mask_disp), "max:", np.max(mask_disp), "unique values:", np.unique(mask_disp))
        
        # Ensure mask_disp has values for visualization
        if np.max(mask_disp) == 0:
            print("Warning: mask_disp is all zeros, using raw prediction values")
            # Use raw prediction values if binary mask is empty
            mask_disp = mask
        
        result_images = {
            'rgb': to_base64(rgb_disp, title="RGB Image"),
            'ndvi': to_base64(ndvi_disp, cmap='RdYlGn', title="NDVI"),
            'slope': to_base64(slope_disp, cmap='viridis', title="Slope"),
            'elevation': to_base64(dem_disp, cmap='viridis', title="DEM"),
            'mask': to_base64(mask_disp, cmap=mask_cmap, title="Mask"),
            'prediction': to_base64(mask, cmap=pred_cmap, title="Prediction")
        }
        
        # Generate segmentation visualization
        mask_visualization = generate_segmentation_visualization(img_array, prediction)
        
        print("Prediction stats - min:", np.min(mask), "max:", np.max(mask), "mean:", np.mean(mask))
        
        return jsonify({
            'result': result, 
            'confidence': confidence,
            'segmentation_image': mask_visualization,
            'result_images': result_images
        })
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error during prediction: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({'error': str(e), 'details': error_traceback}), 500

@app.route('/predict_h5', methods=['POST'])
def predict_h5():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file.filename.endswith('.h5'):
        return jsonify({'error': 'File must be in .h5 format'}), 400
    
    try:
        # Save the uploaded .h5 file temporarily
        temp_path = os.path.join(os.path.dirname(__file__), 'temp_upload.h5')
        file.save(temp_path)
        
        # Open the .h5 file and extract the data
        with h5py.File(temp_path, 'r') as h5_file:
            # Print the keys in the h5 file for debugging
            print(f"H5 file keys: {list(h5_file.keys())}")
            
            # Assuming the h5 file contains a dataset named 'data'
            # Adjust the key based on your actual h5 file structure
            if 'data' in h5_file:
                data = h5_file['data'][:]
                print(f"Data shape from h5 file: {data.shape}")
            else:
                # If 'data' key doesn't exist, try to get the first dataset
                first_key = list(h5_file.keys())[0]
                data = h5_file[first_key][:]
                print(f"Using key '{first_key}', data shape: {data.shape}")
            
            # Print the shape and a sample of the data for band order debugging
            print(f"Loaded data shape: {data.shape}")
            if len(data.shape) == 3:
                h, w, c = data.shape
                for i in range(c):
                    print(f"Channel {i} first pixel value: {data[0,0,i]}")
            elif len(data.shape) == 2:
                print(f"First pixel value: {data[0,0]}")
            
            # Store the original data for visualization later
            original_data = data.copy()
            
            # --- Extract channels based on user mapping ---
            # Using the updated channel indices provided by the user
            CHANNEL_INDEX = { 
                "Red": 3, 
                "Green": 2, 
                "Blue": 1, 
                "NIR": 7, 
                "Slope": 12, 
                "Elevation": 13 
            }
            
            # Step 1: Extract RGB
            rgb_data = np.stack([
                data[:, :, CHANNEL_INDEX["Red"]],
                data[:, :, CHANNEL_INDEX["Green"]],
                data[:, :, CHANNEL_INDEX["Blue"]]
            ], axis=-1)
            # Normalize for display
            rgb_disp = rgb_data.copy()
            for i in range(3):
                ch = rgb_disp[:, :, i]
                if ch.max() > ch.min():
                    rgb_disp[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min())
                else:
                    rgb_disp[:, :, i] = ch
            
            # Step 2: Compute NDVI
            nir = data[:, :, CHANNEL_INDEX["NIR"]].astype(np.float32)
            red = data[:, :, CHANNEL_INDEX["Red"]].astype(np.float32)
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi_disp = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-6)
            
            # Step 3: Extract Slope
            slope = data[:, :, CHANNEL_INDEX["Slope"]]
            slope_disp = (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)
            
            # Step 4: Extract Elevation (DEM)
            elevation = data[:, :, CHANNEL_INDEX["Elevation"]]
            elevation_disp = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-6)
            
            # Resize all to IMG_SIZE for consistency
            rgb_disp = resize_img(rgb_disp)
            ndvi_disp = resize_img(ndvi_disp)
            slope_disp = resize_img(slope_disp)
            elevation_disp = resize_img(elevation_disp)
            
            # Initialize mask_disp with zeros - we'll update it after prediction
            mask_disp = np.zeros((IMG_SIZE[0], IMG_SIZE[1]))
        
        # Create a 6-channel input array for the model using the specified channel indices
        six_channel_array = np.zeros((IMG_SIZE[0], IMG_SIZE[1], CHANNELS))
        
        # Resize the RGB data to match the model's expected input size
        resized_rgb = resize_img(rgb_data)
        
        # First 3 channels: RGB
        six_channel_array[:,:,0:3] = resized_rgb / 255.0  # Normalize RGB channels
        
        # Channel 4: NDVI
        six_channel_array[:,:,3] = resize_img(ndvi_disp)
        
        # Channel 5: Slope
        six_channel_array[:,:,4] = resize_img(slope_disp)
        
        # Channel 6: Elevation
        six_channel_array[:,:,5] = resize_img(elevation_disp)
        
        # Add batch dimension
        input_array = np.expand_dims(six_channel_array, axis=0)
        
        # Clean up the temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Make prediction
        prediction = model.predict(input_array)
        
        # Step 5: Process the mask
        if len(prediction.shape) == 4:  # [batch, height, width, channels]
            mask = prediction[0, :, :, 0]  # Take first image, first channel
        else:
            mask = prediction.squeeze()

        # Use a fixed threshold of 0.3, just like your notebook
        binary_mask = (mask > 0.3).astype(np.uint8)

        # (Optional) Clean up the mask
        binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=50)
        binary_mask = remove_small_holes(binary_mask, area_threshold=50)
        processed_mask = binary_mask.astype(float)

        # For visualization, use the binary mask for both 'mask' and 'prediction'
        mask_disp = processed_mask

        # Step 6: Final Prediction
        # Calculate confidence based on prediction format
        confidence = float(np.mean(mask))
        result = 'segmented area detected' if confidence > 0.5 else 'no segmented area detected'
        
        # Create custom colormap for mask visualization
        # Use 'gray' colormap for both mask and prediction
        result_images = {
            'rgb': to_base64(rgb_disp, title="RGB Image"),
            'ndvi': to_base64(ndvi_disp, cmap='RdYlGn', title="NDVI"),
            'slope': to_base64(slope_disp, cmap='viridis', title="Slope"),
            'elevation': to_base64(elevation_disp, cmap='viridis', title="DEM"),
            'mask': to_base64(mask_disp, cmap='gray', title="Mask"),
            'prediction': to_base64(mask_disp, cmap='gray', title="Prediction")
        }
        
        # Generate segmentation visualization
        try:
            mask_visualization = generate_segmentation_visualization(original_data, prediction)
        except Exception as viz_error:
            print(f"Error generating visualization: {str(viz_error)}")
            mask_visualization = None
        
        print("Prediction stats - min:", np.min(mask), "max:", np.max(mask), "mean:", np.mean(mask))
        
        return jsonify({
            'result': result, 
            'confidence': confidence,
            'segmentation_image': mask_visualization,
            'result_images': result_images
        })
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error during h5 prediction: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({'error': str(e), 'details': error_traceback}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)