import React, { useState } from 'react';

interface ResultImages {
  rgb: string;
  ndvi: string;
  slope: string;
  elevation: string;
  mask: string;
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
      <p className="text-center text-gray-600 mb-4">Upload an image or .h5 file for prediction</p>
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
        <div className="mt-4 text-center">
          <span className={`text-lg font-semibold ${result === 'landslide' ? 'text-red-600' : 'text-green-600'}`}>{result.toUpperCase()}</span>
          {confidence !== null && (
            <div className="text-sm text-gray-500">Confidence: {(confidence * 100).toFixed(2)}%</div>
          )}
        </div>
      )}
      {segmentationImage && (
        <div className="mt-4">
          <h3 className="text-md font-semibold mb-2">Landslide Segmentation:</h3>
          <img 
            src={`data:image/png;base64,${segmentationImage}`} 
            alt="Landslide Segmentation" 
            className="w-full rounded border border-gray-300" 
          />
          <p className="text-xs text-gray-500 mt-1">The highlighted areas indicate potential landslide regions</p>
        </div>
      )}
      
      {resultImages && (
        <div className="mt-4">
          <h3 className="text-md font-semibold mb-2">Multi-Channel Analysis:</h3>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <h4 className="text-sm font-medium">RGB</h4>
              <img 
                src={`data:image/png;base64,${resultImages.rgb}`} 
                alt="RGB" 
                className="w-full rounded border border-gray-300" 
              />
            </div>
            <div>
              <h4 className="text-sm font-medium">NDVI</h4>
              <img 
                src={`data:image/png;base64,${resultImages.ndvi}`} 
                alt="NDVI" 
                className="w-full rounded border border-gray-300" 
              />
            </div>
            <div>
              <h4 className="text-sm font-medium">Slope</h4>
              <img 
                src={`data:image/png;base64,${resultImages.slope}`} 
                alt="Slope" 
                className="w-full rounded border border-gray-300" 
              />
            </div>
            <div>
              <h4 className="text-sm font-medium">Elevation</h4>
              <img 
                src={`data:image/png;base64,${resultImages.elevation}`} 
                alt="Elevation" 
                className="w-full rounded border border-gray-300" 
              />
            </div>
            <div className="col-span-2">
              <h4 className="text-sm font-medium">Landslide Mask</h4>
              <img 
                src={`data:image/png;base64,${resultImages.mask}`} 
                alt="Mask" 
                className="w-full rounded border border-gray-300" 
              />
              <p className="text-xs text-gray-500 mt-1">The highlighted areas indicate potential landslide regions</p>
            </div>
          </div>
        </div>
      )}
      {error && <div className="mt-4 text-center text-red-500">{error}</div>}
    </div>
  );
};

export default ImageUpload;