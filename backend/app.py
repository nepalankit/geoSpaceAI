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

# Import flood prediction module
from flood_prediction import predict_flood, load_flood_model

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
                
                # Use a moderate threshold for visualization that balances visibility with accuracy
                print("Using moderate threshold for visualization to reduce false positives")
                thresh = 0.1  # Moderate threshold to balance visibility with accuracy
                
                # If the prediction values are very high, use a higher threshold for better accuracy
                if np.max(prob_mask_smooth) > 0.5:
                    thresh = 0.15  # Higher threshold for stronger signals
                
                print(f"Visualization - using threshold: {thresh}")
                
                # Create binary mask
                binary_mask = (prob_mask_smooth > thresh).astype(np.uint8)
                
                # Apply morphological operations to clean up and enhance the mask
                from scipy.ndimage import binary_dilation, binary_closing
                
                # First close small gaps
                binary_mask = binary_closing(binary_mask.astype(bool), structure=np.ones((3,3))).astype(bool)
                
                # Then remove small objects and holes
                binary_mask = remove_small_objects(binary_mask, min_size=5)
                binary_mask = remove_small_holes(binary_mask, area_threshold=5)
                
                # Finally dilate to make features more visible
                binary_mask = binary_dilation(binary_mask, iterations=3).astype(np.uint8)
                
                # Print detailed mask statistics for debugging
                print(f"Visualization mask - sum: {np.sum(binary_mask)}, unique values: {np.unique(binary_mask)}")
                print(f"Mask percentage: {np.sum(binary_mask)/(binary_mask.shape[0]*binary_mask.shape[1])*100:.2f}% of image")
                
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
                    # Display the binary mask with a black and white colormap to match the example
                    from matplotlib.colors import ListedColormap
                    bw_cmap = ListedColormap(['black', 'white'])  # Black background, white landslide areas
                    plt.imshow(binary_mask, cmap=bw_cmap, vmin=0, vmax=1)
                    
                    # Add a title that indicates this is the segmentation result
                    plt.title('Segmentation Result')
                    
                    # Add a colorbar for reference
                    divider = make_axes_locatable(plt.gca())
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=bw_cmap), cax=cax)
                    
                    # Add text to explain the visualization
                    if np.sum(binary_mask) > 0:
                        plt.text(0.5, 0.05, f'Landslide detected ({np.sum(binary_mask)} pixels)', 
                                 horizontalalignment='center', verticalalignment='center',
                                 transform=plt.gca().transAxes, color='white', fontsize=10,
                                 bbox=dict(facecolor='black', alpha=0.5))
                
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
        # For mask, use black and white colormap with fixed range
        if title and ('Mask' in title or 'Prediction' in title):
            # For mask, use black and white colormap with fixed range
            if 'Mask' in title:
                # Ensure binary values (0 and 1)
                if img.max() > 0:
                    # Normalize to 0-1 range if not already
                    if img.max() > 1:
                        img = img / img.max()
                    # Show strictly binary mask
                    binary_img = (img > 0).astype(float)
                    from matplotlib.colors import ListedColormap
                    bw_cmap = ListedColormap(['black', 'white'])
                    plt.imshow(binary_img, cmap=bw_cmap, vmin=0, vmax=1)
                else:
                    plt.imshow(np.zeros_like(img), cmap='gray')
                    plt.text(0.5, 0.5, 'No landslide detected', horizontalalignment='center', 
                             verticalalignment='center', transform=plt.gca().transAxes, color='white')
            # For prediction, overlay the mask on the original image for clarity
            elif 'Prediction' in title:
                # If original image is available, overlay
                if img.max() > 0:
                    # Normalize mask
                    if img.max() > 1:
                        img = img / img.max()
                    # Use a strong colormap for mask overlay
                    plt.imshow(img, cmap='inferno', vmin=0, vmax=1, alpha=0.9)
                else:
                    plt.imshow(np.zeros_like(img), cmap='gray')
                    plt.text(0.5, 0.5, 'No landslide predicted', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='white')
        else:
            # For other 2D arrays (like NDVI, slope, etc.)
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
        
        # Calculate NDVI with proper handling of potential NaN values
        ndvi = np.divide(simulated_nir - red, simulated_nir + red + 1e-6)  # Avoid division by zero
        
        # Handle any NaN values that might occur
        ndvi = np.nan_to_num(ndvi, nan=0.0)
        
        # Normalize for display
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
        
        # Match the channel order used in training (from Landslide_detection.ipynb)
        # In training: RED, GREEN, BLUE, NDVI, SLOPE, ELEVATION with 1 - normalization for RGB/SLOPE/ELEVATION
        mid_rgb = np.max(rgb_array) / 2.0  # Similar to training normalization
        mid_slope = np.max(slope_disp) / 2.0
        mid_dem = np.max(dem_disp) / 2.0
        
        # Channel 0: RED (inverted as in training)
        six_channel_array[:,:,0] = 1 - rgb_array[:,:,0] / mid_rgb
        
        # Channel 1: GREEN (inverted as in training)
        six_channel_array[:,:,1] = 1 - rgb_array[:,:,1] / mid_rgb
        
        # Channel 2: BLUE (inverted as in training)
        six_channel_array[:,:,2] = 1 - rgb_array[:,:,2] / mid_rgb
        
        # Channel 3: NDVI
        six_channel_array[:,:,3] = ndvi_disp
        
        # Channel 4: SLOPE (inverted as in training)
        six_channel_array[:,:,4] = 1 - slope_disp / mid_slope
        
        # Channel 5: DEM (inverted as in training)
        six_channel_array[:,:,5] = 1 - dem_disp / mid_dem
        
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
        
        # Use a higher threshold to reduce false positives on non-landslide images
        # For actual prediction (not just visualization)
        # 1. Lower threshold to catch weaker segments
        threshold = 0.1  # Lower threshold for landslide segmentation
        print(f"Using threshold: {threshold}")
        print(f"Raw mask stats: min={np.min(mask)}, max={np.max(mask)}, mean={np.mean(mask)}, std={np.std(mask)}")
        binary_mask = (mask > threshold).astype(np.uint8)

        # 2. (TEMP) Disable small object and hole removal for debugging
        # binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=30)  # Remove smaller objects
        # binary_mask = remove_small_holes(binary_mask, area_threshold=30)  # Remove smaller holes

        # 3. Apply limited morphological closing to connect close regions, but avoid over-expansion
        from scipy.ndimage import binary_closing, label
        binary_mask = binary_closing(binary_mask, structure=np.ones((3,3)), iterations=1)
        binary_mask = binary_mask.astype(np.uint8)

        # 4. Use connected component labeling to highlight segments
        labeled_mask, num_features = label(binary_mask)
        print(f"Number of landslide segments detected: {num_features}")
        # For visualization: strictly binary (0 for background, 1 for landslide area)
        processed_mask = (labeled_mask > 0).astype(float)

        # Print detailed mask statistics for debugging
        print(f"Processed mask - sum: {np.sum(processed_mask)}, unique values: {np.unique(processed_mask)}")
        print(f"Mask percentage: {np.sum(processed_mask)/(processed_mask.shape[0]*processed_mask.shape[1])*100:.2f}% of image")
        
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
        if np.sum(mask > 0.25) > 0:  # Higher threshold for confidence calculation
            confidence = float(np.mean(mask[mask > 0.25]))
        else:
            confidence = float(np.mean(mask))
            
        # Calculate percentage of image covered by the mask
        mask_percentage = np.sum(processed_mask) / (processed_mask.shape[0] * processed_mask.shape[1])
        
        # Adjust result text based on confidence, mask content, and coverage percentage
        # More stringent criteria for landslide detection
        if np.sum(processed_mask) > 50 and confidence > 0.35:  # More pixels required and higher confidence
            result = 'segmented area detected'
        else:
            # Check if there are any areas with high probability
            if np.sum(mask > 0.3) > 20:  # Higher threshold and more pixels
                result = 'potential segmented area detected (low confidence)'
                confidence = max(confidence, 0.3)  # Ensure minimum confidence for UI
            else:
                result = 'no segmented area detected'
                confidence = min(confidence, 0.2)  # Lower confidence for negative results
                
        print(f"Final result: {result}, confidence: {confidence}")
        
        # Ensure confidence is in a reasonable range for UI display
        confidence = max(min(confidence, 0.99), 0.01)  # Clamp between 0.01 and 0.99
        
        # Create custom colormap for mask visualization
        # Use custom colormaps for better visualization
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create a custom colormap for the mask (black and white)
        mask_cmap = LinearSegmentedColormap.from_list('mask_cmap', ['black', 'white'])
        
        # Create a custom colormap for the prediction (purple to yellow)
        pred_cmap = LinearSegmentedColormap.from_list('pred_cmap', ['#2D004B', '#FFFF00'])
        
        # Print mask display stats
        print("Mask display stats - min:", np.min(mask_disp), "max:", np.max(mask_disp), "unique values:", np.unique(mask_disp))
        
        # Ensure mask_disp has values for visualization
        if np.max(mask_disp) == 0:
            print("Warning: mask_disp is all zeros, using raw prediction values")
            # Use raw prediction values if binary mask is empty
            mask_disp = mask
        
        # Ensure the mask and prediction have visible values
        # For binary mask, use white for landslide areas
        binary_mask_display = binary_mask.astype(float)
        
        # For prediction, use the raw probability values with a colormap
        prediction_display = mask
        
        result_images = {
            'rgb': to_base64(rgb_disp, title="RGB Image"),
            'ndvi': to_base64(ndvi_disp, cmap='RdYlGn', title="NDVI"),
            'slope': to_base64(slope_disp, cmap='viridis', title="Slope"),
            'elevation': to_base64(dem_disp, cmap='viridis', title="DEM"),
            'mask': to_base64(binary_mask_display, cmap=mask_cmap, title="Mask"),
            'prediction': to_base64(prediction_display, cmap=pred_cmap, title="Prediction")
        }
        
        # Generate segmentation visualization
        try:
            # Generate the full visualization with multiple panels
            full_visualization = generate_segmentation_visualization(img_array, prediction)
            
            # Generate a simple black and white mask visualization showing only the landslide segments
            # Create a figure with just the binary mask
            plt.figure(figsize=(5, 5))
            
            # Create binary mask from prediction
            if len(prediction.shape) == 4:  # Batch, height, width, channels
                mask_for_viz = prediction[0, :, :, 0]  # Take first batch item, first channel
            elif len(prediction.shape) == 3 and prediction.shape[2] > 1:  # Multi-class
                mask_for_viz = prediction[:, :, 0]  # Take first channel
            else:  # Binary segmentation
                mask_for_viz = prediction.squeeze()
            
            # Apply threshold to create binary mask
            binary_mask_viz = (mask_for_viz > 0.15).astype(np.uint8)  # Use same threshold as in prediction
            
            # Apply morphological operations to clean up the mask
            from scipy.ndimage import binary_closing, binary_dilation
            binary_mask_viz = binary_closing(binary_mask_viz.astype(bool), structure=np.ones((3,3))).astype(bool)
            binary_mask_viz = remove_small_objects(binary_mask_viz, min_size=5)
            binary_mask_viz = remove_small_holes(binary_mask_viz, area_threshold=5)
            binary_mask_viz = binary_dilation(binary_mask_viz, iterations=2).astype(np.uint8)
            
            # Create a black and white colormap
            from matplotlib.colors import ListedColormap
            bw_cmap = ListedColormap(['black', 'white'])  # Black background, white landslide areas
            
            # Display the binary mask with black background and white landslide areas
            plt.imshow(binary_mask_viz, cmap=bw_cmap, vmin=0, vmax=1)
            plt.axis('off')  # Turn off axis
            
            # Remove all margins and padding
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            
            # Save the binary mask visualization to a base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            buffer.seek(0)
            binary_mask_visualization = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Use the binary mask visualization as the segmentation image
            mask_visualization = binary_mask_visualization
        except Exception as viz_error:
            print(f"Error generating visualization: {str(viz_error)}")
            mask_visualization = None
        
        print("Prediction stats - min:", np.min(mask), "max:", np.max(mask), "mean:", np.mean(mask))
        
        # Prepare the response with clear indication for non-landslide images
        response_data = {
            'result': result, 
            'confidence': confidence,
            'segmentation_image': mask_visualization,
            'result_images': result_images,
            'is_landslide': result == 'segmented area detected',  # Boolean flag for frontend
            'message': ''
        }
        
        # Add a clear message for non-landslide images
        if result == 'no segmented area detected':
            response_data['message'] = 'No landslide detected in this image.'
        elif 'low confidence' in result:
            response_data['message'] = 'Potential landslide detected with low confidence. Please verify.'
        else:
            response_data['message'] = 'Landslide detected in this image.'
            
        return jsonify(response_data)
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
            # Handle any potential NaN values
            ndvi = np.nan_to_num(ndvi, nan=0.0)
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
        
        # Match the channel order used in training (from Landslide_detection.ipynb)
        # In training: RED, GREEN, BLUE, NDVI, SLOPE, ELEVATION with 1 - normalization for RGB/SLOPE/ELEVATION
        mid_rgb = np.max(resized_rgb) / 2.0  # Similar to training normalization
        mid_slope = np.max(slope_disp) / 2.0
        mid_dem = np.max(elevation_disp) / 2.0
        
        # Channel 0: RED (inverted as in training)
        six_channel_array[:,:,0] = 1 - resized_rgb[:,:,0] / mid_rgb
        
        # Channel 1: GREEN (inverted as in training)
        six_channel_array[:,:,1] = 1 - resized_rgb[:,:,1] / mid_rgb
        
        # Channel 2: BLUE (inverted as in training)
        six_channel_array[:,:,2] = 1 - resized_rgb[:,:,2] / mid_rgb
        
        # Channel 3: NDVI
        six_channel_array[:,:,3] = resize_img(ndvi_disp)
        
        # Channel 4: SLOPE (inverted as in training)
        six_channel_array[:,:,4] = 1 - resize_img(slope_disp) / mid_slope
        
        # Channel 5: DEM (inverted as in training)
        six_channel_array[:,:,5] = 1 - resize_img(elevation_disp) / mid_dem
        
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

        # Use a higher threshold to reduce false positives on non-landslide images
        # For actual prediction (not just visualization)
        threshold = 0.25  # Higher threshold to only show confident landslide areas
        print(f"Using threshold: {threshold}")
        
        binary_mask = (mask > threshold).astype(np.uint8)

        # Apply morphological operations to clean up and enhance the mask
        from scipy.ndimage import binary_dilation, binary_closing
        
        # First close small gaps
        binary_mask = binary_closing(binary_mask.astype(bool), structure=np.ones((3,3))).astype(bool)
        
        # Then remove small objects and holes
        binary_mask = remove_small_objects(binary_mask, min_size=5)
        binary_mask = remove_small_holes(binary_mask, area_threshold=5)
        
        # Finally dilate to make features more visible
        binary_mask = binary_dilation(binary_mask, iterations=3).astype(np.uint8)
        processed_mask = binary_mask.astype(float)
        
        # Print detailed mask statistics for debugging
        print(f"Processed mask - sum: {np.sum(processed_mask)}, unique values: {np.unique(processed_mask)}")
        print(f"Mask percentage: {np.sum(processed_mask)/(processed_mask.shape[0]*processed_mask.shape[1])*100:.2f}% of image")

        # For visualization, use the binary mask for both 'mask' and 'prediction'
        mask_disp = processed_mask

        # Step 6: Final Prediction
        # Calculate confidence as mean probability of segmented class
        # If the mask has any positive values, use their mean as confidence
        if np.sum(mask > 0.2) > 0:  # Higher threshold for confidence calculation
            confidence = float(np.mean(mask[mask > 0.2]))
        else:
            confidence = float(np.mean(mask))
            
        # Calculate percentage of image covered by the mask
        mask_percentage = np.sum(processed_mask) / (processed_mask.shape[0] * processed_mask.shape[1])
        
        # Adjust result text based on confidence, mask content, and coverage percentage
        # More stringent criteria for landslide detection
        if np.sum(processed_mask) > 50 and confidence > 0.3:  # More pixels required and higher confidence
            result = 'segmented area detected'
        else:
            # Check if there are any areas with high probability
            if np.sum(mask > 0.25) > 20:  # Higher threshold and more pixels
                result = 'potential segmented area detected (low confidence)'
                confidence = max(confidence, 0.3)  # Ensure minimum confidence for UI
            else:
                result = 'no segmented area detected'
                confidence = min(confidence, 0.2)  # Lower confidence for negative results
                
        print(f"Final result: {result}, confidence: {confidence}")
        
        # Create custom colormap for mask visualization
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create a custom colormap for the mask (black and white)
        mask_cmap = LinearSegmentedColormap.from_list('mask_cmap', ['black', 'white'])
        
        # Create a custom colormap for the prediction (purple to yellow)
        pred_cmap = LinearSegmentedColormap.from_list('pred_cmap', ['#2D004B', '#FFFF00'])
        
        # Ensure the mask and prediction have visible values
        # For binary mask, use white for landslide areas
        binary_mask_display = binary_mask.astype(float)
        
        # For prediction, use the raw probability values with a colormap
        prediction_display = mask
        
        result_images = {
            'rgb': to_base64(rgb_disp, title="RGB Image"),
            'ndvi': to_base64(ndvi_disp, cmap='RdYlGn', title="NDVI"),
            'slope': to_base64(slope_disp, cmap='viridis', title="Slope"),
            'elevation': to_base64(elevation_disp, cmap='viridis', title="DEM"),
            'mask': to_base64(binary_mask_display, cmap=mask_cmap, title="Mask"),
            'prediction': to_base64(prediction_display, cmap=pred_cmap, title="Prediction")
        }
        
        # Generate segmentation visualization
        try:
            # Create a figure with just the binary mask - showing ONLY landslide areas
            plt.figure(figsize=(5, 5))
            
            # Create binary mask from prediction
            if len(prediction.shape) == 4:  # Batch, height, width, channels
                mask = prediction[0, :, :, 0]  # Take first batch item, first channel
            elif len(prediction.shape) == 3 and prediction.shape[2] > 1:  # Multi-class
                mask = prediction[:, :, 0]  # Take first channel
            else:  # Binary segmentation
                mask = prediction.squeeze()
            
            # Apply threshold to create binary mask - use higher threshold to only show confident landslide areas
            binary_mask = (mask > 0.25).astype(np.uint8)  # Higher threshold to only show confident landslide areas
            
            # Apply morphological operations to clean up the mask
            from scipy.ndimage import binary_closing, binary_dilation
            binary_mask = binary_closing(binary_mask.astype(bool), structure=np.ones((3,3))).astype(bool)
            binary_mask = remove_small_objects(binary_mask, min_size=10)  # Remove smaller objects
            binary_mask = remove_small_holes(binary_mask, area_threshold=10)  # Remove smaller holes
            binary_mask = binary_dilation(binary_mask, iterations=1).astype(np.uint8)  # Less dilation to avoid over-expansion
            
            # Create a black and white colormap
            from matplotlib.colors import ListedColormap
            bw_cmap = ListedColormap(['black', 'white'])  # Black background, white landslide areas
            
            # Display the binary mask with black background and white landslide areas
            plt.imshow(binary_mask, cmap=bw_cmap, vmin=0, vmax=1)
            plt.axis('off')  # Turn off axis
            
            # Remove all margins and padding
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            
            # Save the binary mask visualization to a base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            buffer.seek(0)
            binary_mask_visualization = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Use the binary mask visualization as the segmentation image
            mask_visualization = binary_mask_visualization
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

# Add flood prediction endpoint
@app.route('/predict_flood', methods=['POST'])
def flood_prediction_endpoint():
    return predict_flood()

if __name__ == '__main__':
    # Load flood model on startup
    flood_model = load_flood_model()
    app.run(host='0.0.0.0', port=5001, debug=True)