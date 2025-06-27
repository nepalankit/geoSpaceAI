import React, { useState } from 'react';

const ModelInfo: React.FC = () => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="mt-6 bg-white rounded-lg shadow-md p-4 border border-gray-200">
      <div className="flex justify-between items-center cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
        <h3 className="text-md font-semibold">About Landslide Detection Model</h3>
        <button className="text-blue-600 hover:text-blue-800">
          {isExpanded ? 'Hide Details' : 'Show Details'}
        </button>
      </div>
      
      {isExpanded && (
        <div className="mt-4 text-sm text-gray-700 space-y-3">
          <p>
            This landslide detection system uses a U-Net architecture, a specialized convolutional neural network 
            designed for biomedical image segmentation that has proven effective for remote sensing applications.
          </p>
          
          <h4 className="font-medium text-gray-900 mt-2">Input Data:</h4>
          <ul className="list-disc pl-5 space-y-1">
            <li><span className="font-medium">RGB Bands:</span> Visual light information from satellite imagery</li>
            <li><span className="font-medium">NDVI:</span> Normalized Difference Vegetation Index - indicates vegetation health</li>
            <li><span className="font-medium">Slope:</span> Terrain steepness derived from elevation data</li>
            <li><span className="font-medium">DEM:</span> Digital Elevation Model - terrain height information</li>
          </ul>
          
          <h4 className="font-medium text-gray-900 mt-2">How It Works:</h4>
          <ol className="list-decimal pl-5 space-y-1">
            <li>The model processes multi-channel input (RGB + NDVI + Slope + DEM)</li>
            <li>The U-Net architecture extracts features at multiple scales</li>
            <li>The decoder part reconstructs a probability map for landslide presence</li>
            <li>Post-processing applies thresholding and morphological operations</li>
            <li>The final output highlights areas with high landslide probability</li>
          </ol>
          
          <div className="bg-blue-50 p-3 rounded mt-2">
            <p className="text-blue-800">
              <span className="font-medium">Note:</span> For optimal results, use .h5 files containing all required 
              channels. When using regular images, the system simulates NDVI, slope, and elevation data from RGB values, 
              which may reduce accuracy.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelInfo;