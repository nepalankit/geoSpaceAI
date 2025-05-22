
import { Card, CardContent } from "@/components/ui/card";
import { featuresData } from "@/data/sampleData";

const FeatureSection = () => {
  const getIconByName = (name: string) => {
    switch (name) {
      case "flame":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="w-10 h-10"
          >
            <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"></path>
          </svg>
        );
      case "drop":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="w-10 h-10"
          >
            <path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"></path>
          </svg>
        );
      case "tree":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="w-10 h-10"
          >
            <path d="M17 14v7m-5-7v7M8 14v7M3 9a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-.5a6 6 0 0 0-6-6 6 6 0 0 0-6 6A6 6 0 0 0 3 9.5z" />
          </svg>
        );
      case "mountain":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="w-10 h-10"
          >
            <path d="m8 3 4 8 5-5 5 15H2L8 3z"></path>
          </svg>
        );
      default:
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="w-10 h-10"
          >
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="2" y1="12" x2="22" y2="12"></line>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
          </svg>
        );
    }
  };

  const getColorByName = (name: string) => {
    switch (name) {
      case "fire":
        return "text-fire-600";
      case "water":
        return "text-water-600";
      case "forest":
        return "text-forest-600";
      case "earth":
        return "text-earth-600";
      default:
        return "text-primary";
    }
  };

  return (
    <section id="features" className="py-12 md:py-24">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col gap-4 items-center text-center mb-10 md:mb-16">
          <div className="inline-flex items-center justify-center px-4 py-1.5 rounded-full border text-sm font-medium mb-2">
            Advanced Capabilities
          </div>
          <h2 className="text-3xl font-bold tracking-tighter">AI-Powered Anomaly Detection</h2>
          <p className="max-w-[700px] text-muted-foreground">
            Our advanced machine learning algorithms analyze satellite and drone imagery to detect environmental anomalies with high precision and reliability.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12">
          {featuresData.map((feature, index) => (
            <Card key={index} className={`bg-muted/40 border-l-4 border-${feature.color}-500 hover:shadow-md transition-shadow overflow-hidden`}>
              <CardContent className="p-6">
                <div className="flex gap-4 items-start">
                  <div className={`shrink-0 ${getColorByName(feature.color)}`}>
                    {getIconByName(feature.icon)}
                  </div>
                  <div>
                    <h3 className="font-semibold text-xl mb-2">{feature.title}</h3>
                    <p className="text-muted-foreground">{feature.description}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-16 p-6 md:p-12 bg-muted rounded-lg border text-center">
          <h3 className="text-2xl font-bold mb-4">How GeoScopeAI Works</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 md:gap-4">
            <div className="flex flex-col items-center">
              <div className="flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 text-primary mb-4">
                <span className="font-bold text-2xl">1</span>
              </div>
              <h4 className="font-medium mb-2">Data Collection</h4>
              <p className="text-sm text-muted-foreground">Gathering satellite and drone imagery from multiple sources</p>
            </div>

            <div className="flex flex-col items-center">
              <div className="flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 text-primary mb-4">
                <span className="font-bold text-2xl">2</span>
              </div>
              <h4 className="font-medium mb-2">AI Analysis</h4>
              <p className="text-sm text-muted-foreground">Processing imagery through our advanced machine learning algorithms</p>
            </div>

            <div className="flex flex-col items-center">
              <div className="flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 text-primary mb-4">
                <span className="font-bold text-2xl">3</span>
              </div>
              <h4 className="font-medium mb-2">Anomaly Detection</h4>
              <p className="text-sm text-muted-foreground">Identifying potential environmental anomalies and threats</p>
            </div>

            <div className="flex flex-col items-center">
              <div className="flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 text-primary mb-4">
                <span className="font-bold text-2xl">4</span>
              </div>
              <h4 className="font-medium mb-2">Alert Generation</h4>
              <p className="text-sm text-muted-foreground">Creating actionable alerts for response teams and stakeholders</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default FeatureSection;
