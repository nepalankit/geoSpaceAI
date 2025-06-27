import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Dashboard from "@/components/Dashboard";
import FeatureSection from "@/components/FeatureSection";
import Footer from "@/components/Footer";
import ImageUpload from "@/components/ImageUpload";
import { Link } from "react-router-dom";

const Index = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main>
        <section id="home">
          <Hero />
        </section>
        <section id="image-upload">
          <div className="container mx-auto px-4 py-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h2 className="text-2xl font-bold mb-4 text-center">Landslide Detection</h2>
                <ImageUpload />
              </div>
              <div>
                <h2 className="text-2xl font-bold mb-4 text-center">Flood Detection</h2>
                <div className="p-6 bg-white rounded-lg shadow-md">
                  <p className="text-center text-gray-600 mb-4">Try our dedicated flood detection tool with advanced visualization and analysis features.</p>
                  <Link to="/flood-detection" className="block w-full py-2 px-4 bg-blue-600 text-white text-center rounded hover:bg-blue-700 transition-colors">
                    Go to Flood Detection
                  </Link>
                  <div className="mt-4 p-4 bg-blue-50 rounded-md">
                    <p className="text-sm text-blue-800">Our flood detection model uses a U-Net architecture to identify flooded areas in satellite or aerial imagery with high accuracy.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
        <section id="dashboard">
          <Dashboard />
        </section>
        <section id="features">
          <FeatureSection />
        </section>
        <section id="about" className="py-16 bg-muted/50">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center text-center space-y-4">
              <h2 className="text-3xl font-bold tracking-tighter">About GeoScopeAI</h2>
              <p className="text-muted-foreground max-w-[700px]">
                GeoScopeAI is an innovative platform that uses advanced AI algorithms to detect anomalies 
                such as wildfires, floods, illegal deforestation, and land degradation in satellite and UAV drone images. 
                Our mission is to automate the monitoring of large natural areas, enabling faster response to 
                environmental threats and improving ecological conservation efforts.
              </p>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
};

export default Index;
