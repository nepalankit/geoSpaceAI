import React, { useState, useEffect } from 'react';

interface ResultImages {
  rgb: string;
  mask: string;
  prediction: string;
}

const FloodSegmentation: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [segmentationImage, setSegmentationImage] = useState<string | null>(null);
  const [resultImages, setResultImages] = useState<ResultImages | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setSelectedFile(file);
    setResult(null);
    setConfidence(null);
    setError(null);
    
    if (file) {
      setPreview(URL.createObjectURL(file));
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
      
      // Call the flood segmentation endpoint
      const response = await fetch('http://localhost:5001/predict_flood', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      if (response.ok) {
        setResult(data.result);
        setConfidence(data.confidence);
        
        // Check if segmentation image is available
        if (data.segmentation_image) {
          setSegmentationImage(data.segmentation_image);
        }
        
        // Handle result_images
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
      <h2 className="text-xl font-bold mb-4 text-center">Flood Area Segmentation</h2>
      <p className="text-center text-gray-600 mb-4">Upload satellite or aerial imagery to detect flooded areas</p>
      <div className="bg-blue-50 p-3 rounded-md mb-4 text-sm">
        <p className="font-medium text-blue-800">For best results:</p>
        <ul className="list-disc pl-5 text-blue-700 mt-1">
          <li>Use high-resolution aerial photos of flood-affected areas</li>
          <li>Images should have clear contrast between water and land</li>
          <li>RGB images work best for this model</li>
        </ul>
      </div>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
        {preview && (
          <img src={preview} alt="Preview" className="w-full h-48 object-cover rounded" />
        )}
        <button
          type="submit"
          disabled={!selectedFile || loading}
          className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Detect Flooded Areas'}
        </button>
      </form>
      {result && (
        <div className="mt-4 p-3 rounded-md border" style={{ borderColor: result.includes('detected') ? '#3b82f6' : '#10b981' }}>
          <div className="text-center">
            <span className={`text-lg font-semibold ${result.includes('detected') ? 'text-blue-600' : 'text-green-600'}`}>
              {result.includes('detected') ? 'FLOOD AREA DETECTED' : 'NO FLOOD AREA DETECTED'}
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
                ? 'Potential flood areas have been identified in this image. Review the segmentation for details.'
                : 'No significant flood features were detected in this image.'}
            </p>
          </div>
        </div>
      )}
      {segmentationImage && (
        <div className="mt-4">
          <h3 className="text-md font-semibold mb-2">Flood Area Segmentation:</h3>
          <img 
            src={`data:image/png;base64,${segmentationImage}`} 
            alt="Segmentation Result" 
            className="w-full rounded border border-gray-300" 
          />
          <p className="text-xs text-gray-500 mt-1">The highlighted areas indicate potential flooded regions</p>
        </div>
      )}
      
      {resultImages && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3">Analysis:</h3>
          <div className="grid grid-cols-2 gap-4">
            {resultImages.rgb && (
              <div className="border rounded-md p-2 bg-gray-50">
                <img 
                  src={`data:image/png;base64,${resultImages.rgb}`} 
                  alt="RGB" 
                  className="w-full rounded border border-gray-300" 
                />
                <p className="text-sm font-medium text-center mt-1">Original Image</p>
                <p className="text-xs text-gray-600 mt-1">Input image showing the area being analyzed</p>
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
                <p className="text-xs text-gray-600 mt-1">Shows areas classified as flooded (white) vs non-flooded (black)</p>
              </div>
            )}
            {resultImages.prediction && (
              <div className="border rounded-md p-2 bg-gray-50 col-span-2">
                <img 
                  src={`data:image/png;base64,${resultImages.prediction}`} 
                  alt="Prediction" 
                  className="w-full rounded border border-gray-300" 
                />
                <p className="text-sm font-medium text-center mt-1">Prediction Overlay</p>
                <p className="text-xs text-gray-600 mt-1">Flood probability map showing likelihood of flooding in different areas</p>
              </div>
            )}
          </div>
          <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3 mt-4">
            <p className="text-sm text-yellow-800">
              <span className="font-medium">How to interpret:</span> The model analyzes the image to identify flooded areas based on 
              water signatures. Highlighted areas in the prediction and mask images indicate potential flood regions. 
              This can help in disaster response planning and damage assessment.
            </p>
          </div>
        </div>
      )}
      {loading && (
        <div className="flex flex-col items-center justify-center mt-4 p-4">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-700 mb-2"></div>
          <p className="text-center text-gray-700">Processing image...</p>
          <p className="text-center text-xs text-gray-500 mt-1">Running flood detection model</p>
        </div>
      )}
      {error && <div className="mt-4 text-center text-red-500">{error}</div>}
      
      <div className="mt-6 p-4 bg-gray-50 rounded-md border border-gray-200">
        <h3 className="text-md font-semibold mb-2">About This Model:</h3>
        <p className="text-sm text-gray-700">
          This flood segmentation model uses a U-Net architecture to identify areas affected by flooding in satellite or aerial imagery. 
          The model was trained on a dataset of 290 images with corresponding flood masks, achieving high accuracy in detecting 
          water-inundated areas.
        </p>
        <p className="text-sm text-gray-700 mt-2">
          The model works by analyzing pixel patterns and spectral characteristics to distinguish between water and non-water areas, 
          making it useful for rapid flood mapping and disaster response.
        </p>
      </div>
    </div>
  );
};

export default FloodSegmentation;