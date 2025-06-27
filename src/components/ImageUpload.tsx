import React, { useState, useEffect } from 'react';
import ModelInfo from './ModelInfo';

interface ResultImages {
  rgb: string;
  ndvi: string;
  slope: string;
  dem: string;
  mask: string;
  prediction: string;
}

const ImageUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileType, setFileType] = useState<'image' | 'h5'>('image');
  const [segmentationImage, setSegmentationImage] = useState<string | null>(null);
  const [resultImages, setResultImages] = useState<ResultImages | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setSelectedFile(file);
    setResult(null);
    setConfidence(null);
    setError(null);
    
    if (file) {
      // Automatically detect file type based on extension
      if (file.name.toLowerCase().endsWith('.h5')) {
        setFileType('h5');
        // No preview for h5 files
        setPreview(null);
      } else {
        setFileType('image');
        setPreview(URL.createObjectURL(file));
      }
    } else {
      setPreview(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;
    setLoading(true);
    setResult(null);
    setConfidence(null);
    setError(null);
    setSegmentationImage(null);
    setResultImages(null);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      // Use the appropriate endpoint based on file type
      const endpoint = fileType === 'h5' ? 'predict_h5' : 'predict';
      const response = await fetch(`http://localhost:5001/${endpoint}`, {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      if (response.ok) {
        setResult(data.result);
        setConfidence(data.confidence);
        
        // Check if segmentation image is available (for h5 files with segmentation masks)
        if (data.segmentation_image) {
          setSegmentationImage(data.segmentation_image);
        }
        
        // Handle result_images for h5 files
        if (data.result_images) {
          setResultImages(data.result_images);
        }
      } else {
        setError(data.error || 'Prediction failed.');
      }
    } catch (err) {
      setError('Failed to connect to backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-white rounded-lg shadow-md mt-8">
      <h2 className="text-xl font-bold mb-4 text-center">Landslide Detection</h2>
      <p className="text-center text-gray-600 mb-4">Upload satellite imagery (.h5 file) or regular image for landslide prediction</p>
      <div className="bg-blue-50 p-3 rounded-md mb-4 text-sm">
        <p className="font-medium text-blue-800">For best results:</p>
        <ul className="list-disc pl-5 text-blue-700 mt-1">
          <li>Use .h5 files with multi-channel satellite data</li>
          <li>Ensure imagery contains RGB, elevation, and slope data</li>
          <li>For regular images, high-resolution aerial photos work best</li>
        </ul>
      </div>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <input
          type="file"
          accept="image/*,.h5"
          onChange={handleFileChange}
          className="file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
        <div className="text-sm text-gray-500 mt-1">
          {fileType === 'image' ? 'Image file selected' : 'H5 file selected'}
        </div>
        {preview && (
          <img src={preview} alt="Preview" className="w-full h-48 object-cover rounded" />
        )}
        <button
          type="submit"
          disabled={!selectedFile || loading}
          className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>
      {result && (
        <div className="mt-4 p-3 rounded-md border" style={{ borderColor: result.includes('detected') ? '#3b82f6' : '#10b981' }}>
          <div className="text-center">
            <span className={`text-lg font-semibold ${result.includes('detected') ? 'text-blue-600' : 'text-green-600'}`}>
              {result.includes('detected') ? 'LANDSLIDE DETECTED' : 'NO LANDSLIDE DETECTED'}
            </span>
            {confidence !== null && (
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className={`h-2.5 rounded-full ${result.includes('detected') ? 'bg-blue-600' : 'bg-green-600'}`} 
                    style={{ width: `${(confidence * 100).toFixed(0)}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0%</span>
                  <span>Confidence: {(confidence * 100).toFixed(2)}%</span>
                  <span>100%</span>
                </div>
              </div>
            )}
            <p className="text-sm mt-2 text-gray-600">
              {result.includes('detected') 
                ? 'Potential landslide areas have been identified in this image. Review the segmentation for details.'
                : 'No significant landslide features were detected in this image.'}
            </p>
          </div>
        </div>
      )}
      {segmentationImage && (
        <div className="mt-4">
          <h3 className="text-md font-semibold mb-2">Landslide Segmentation:</h3>
          <img 
            src={`data:image/png;base64,${segmentationImage}`} 
            alt="Segmentation Result" 
            className="w-full rounded border border-gray-300" 
          />
          <p className="text-xs text-gray-500 mt-1">The highlighted areas indicate potential landslide regions</p>
        </div>
      )}
      
      {resultImages && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3">Multi-Channel Analysis:</h3>
          <div className="grid grid-cols-2 gap-4">
            {resultImages.rgb && (
              <div className="border rounded-md p-2 bg-gray-50">
                <img 
                  src={`data:image/png;base64,${resultImages.rgb}`} 
                  alt="RGB" 
                  className="w-full rounded border border-gray-300" 
                />
                <p className="text-sm font-medium text-center mt-1">RGB</p>
                <p className="text-xs text-gray-600 mt-1">Original satellite imagery showing visible light bands</p>
              </div>
            )}
            {resultImages.ndvi && (
              <div className="border rounded-md p-2 bg-gray-50">
                <img 
                  src={`data:image/png;base64,${resultImages.ndvi}`} 
                  alt="NDVI" 
                  className="w-full rounded border border-gray-300" 
                />
                <p className="text-sm font-medium text-center mt-1">NDVI</p>
                <p className="text-xs text-gray-600 mt-1">Vegetation index - helps identify disturbed vegetation patterns</p>
              </div>
            )}
            {resultImages.slope && (
              <div className="border rounded-md p-2 bg-gray-50">
                <img 
                  src={`data:image/png;base64,${resultImages.slope}`} 
                  alt="Slope" 
                  className="w-full rounded border border-gray-300" 
                />
                <p className="text-sm font-medium text-center mt-1">Slope</p>
                <p className="text-xs text-gray-600 mt-1">Terrain steepness - critical factor in landslide susceptibility</p>
              </div>
            )}
            {resultImages.dem && (
                <div className="border rounded-md p-2 bg-gray-50">
                  <img 
                    src={`data:image/png;base64,${resultImages.dem}`} 
                    alt="Elevation" 
                    className="w-full rounded border border-gray-300" 
                  />
                  <p className="text-sm font-medium text-center mt-1">Elevation</p>
                  <p className="text-xs text-gray-600 mt-1">Digital Elevation Model - shows terrain height information</p>
                </div>
              )}
            {resultImages.mask && (
              <div className="border rounded-md p-2 bg-gray-50">
                <img 
                  src={`data:image/png;base64,${resultImages.mask}`} 
                  alt="Mask" 
                  className="w-full rounded border border-gray-300" 
                />
                <p className="text-sm font-medium text-center mt-1">Binary Mask</p>
                <p className="text-xs text-gray-600 mt-1">Shows areas classified as landslide (white) vs non-landslide (black)</p>
              </div>
            )}
            {resultImages.prediction && (
              <div className="border rounded-md p-2 bg-gray-50">
                <img 
                  src={`data:image/png;base64,${resultImages.prediction}`} 
                  alt="Prediction" 
                  className="w-full rounded border border-gray-300" 
                />
                <p className="text-sm font-medium text-center mt-1">Prediction Overlay</p>
                <p className="text-xs text-gray-600 mt-1">Landslide probability map overlaid on original image</p>
              </div>
            )}
          </div>
          <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3 mt-4">
            <p className="text-sm text-yellow-800">
              <span className="font-medium">How to interpret:</span> The model analyzes multiple data channels to identify landslide features. 
              Highlighted areas in the prediction and mask images indicate potential landslide regions based on terrain characteristics, 
              vegetation patterns, and spectral signatures.
            </p>
          </div>
        </div>
      )}
      {loading && (
        <div className="flex flex-col items-center justify-center mt-4 p-4">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-700 mb-2"></div>
          <p className="text-center text-gray-700">Processing satellite data...</p>
          <p className="text-center text-xs text-gray-500 mt-1">Running landslide detection model</p>
        </div>
      )}
      {error && <div className="mt-4 text-center text-red-500">{error}</div>}
      
      <ModelInfo />
    </div>
  );
};

export default ImageUpload;