import React from 'react';
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import FloodSegmentation from "@/components/FloodSegmentation";

const FloodDetection: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-grow">
        <section className="py-12 bg-gradient-to-b from-blue-50 to-white">
          <div className="container mx-auto px-4">
            <div className="max-w-3xl mx-auto text-center mb-12">
              <h1 className="text-4xl font-bold tracking-tight text-blue-800 mb-4">Flood Detection</h1>
              <p className="text-lg text-gray-600">
                Upload satellite or aerial imagery to detect flooded areas using our advanced AI model.
                This tool helps in disaster response, damage assessment, and environmental monitoring.
              </p>
            </div>
            
            <div className="max-w-3xl mx-auto">
              <FloodSegmentation />
            </div>
          </div>
        </section>
        
        <section className="py-16 bg-white">
          <div className="container mx-auto px-4">
            <div className="max-w-3xl mx-auto">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">About Flood Detection</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
                <div className="bg-blue-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold text-blue-700 mb-3">How It Works</h3>
                  <p className="text-gray-700">
                    Our flood detection model uses a U-Net architecture trained on satellite imagery to identify 
                    water-inundated areas. The model analyzes pixel patterns and spectral characteristics to 
                    distinguish between water and non-water areas with high accuracy.
                  </p>
                </div>
                
                <div className="bg-blue-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold text-blue-700 mb-3">Applications</h3>
                  <ul className="list-disc pl-5 text-gray-700 space-y-2">
                    <li>Rapid flood mapping during disaster response</li>
                    <li>Post-disaster damage assessment</li>
                    <li>Flood risk analysis and planning</li>
                    <li>Environmental monitoring and climate change impact studies</li>
                    <li>Insurance claim verification</li>
                  </ul>
                </div>
              </div>
              
              <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
                <h3 className="text-xl font-semibold text-gray-800 mb-3">Best Practices for Accurate Results</h3>
                <ul className="list-disc pl-5 text-gray-700 space-y-2">
                  <li><span className="font-medium">Image Quality:</span> Use high-resolution aerial or satellite imagery for best results</li>
                  <li><span className="font-medium">Image Timing:</span> Recent imagery captured during or shortly after flooding events works best</li>
                  <li><span className="font-medium">Image Content:</span> Images should have clear contrast between water and land areas</li>
                  <li><span className="font-medium">Format:</span> RGB images in common formats (JPEG, PNG) are supported</li>
                  <li><span className="font-medium">Interpretation:</span> Review both the binary mask and probability map for comprehensive analysis</li>
                </ul>
              </div>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
};

export default FloodDetection;