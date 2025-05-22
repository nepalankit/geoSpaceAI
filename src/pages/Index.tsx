
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Dashboard from "@/components/Dashboard";
import FeatureSection from "@/components/FeatureSection";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main>
        <section id="home">
          <Hero />
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
