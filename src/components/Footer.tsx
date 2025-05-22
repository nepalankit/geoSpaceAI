
import { Link } from "react-router-dom";

const Footer = () => {
  return (
    <footer id="about" className="bg-muted/30 py-12 md:py-16">
      <div className="container px-4 md:px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 md:gap-12">
          <div className="col-span-1 md:col-span-2">
            <Link to="/" className="flex items-center gap-2 mb-4">
              <div className="bg-gradient-to-r from-primary to-secondary rounded-md p-1.5">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  className="w-5 h-5 text-white"
                >
                  <circle cx="12" cy="12" r="10" />
                  <path d="m4.93 4.93 4.24 4.24" />
                  <path d="m14.83 9.17 4.24-4.24" />
                  <path d="m14.83 14.83 4.24 4.24" />
                  <path d="m9.17 14.83-4.24 4.24" />
                </svg>
              </div>
              <span className="font-bold text-xl">GeoScopeAI</span>
            </Link>
            <p className="text-muted-foreground mb-4 max-w-md">
              GeoScopeAI uses advanced artificial intelligence to detect environmental anomalies from satellite and drone imagery, enabling faster response to environmental threats and disasters.
            </p>
            <div className="flex space-x-4">
              <a href="#" className="text-muted-foreground hover:text-foreground">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-5 w-5"
                >
                  <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                  <rect width="4" height="12" x="2" y="9"></rect>
                  <circle cx="4" cy="4" r="2"></circle>
                </svg>
                <span className="sr-only">LinkedIn</span>
              </a>
              <a href="#" className="text-muted-foreground hover:text-foreground">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-5 w-5"
                >
                  <path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z"></path>
                </svg>
                <span className="sr-only">Twitter</span>
              </a>
              <a href="#" className="text-muted-foreground hover:text-foreground">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-5 w-5"
                >
                  <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z"></path>
                </svg>
                <span className="sr-only">Facebook</span>
              </a>
            </div>
          </div>
          
          <div>
            <h3 className="font-medium text-lg mb-4">Product</h3>
            <nav className="flex flex-col space-y-3">
              <Link to="#" className="text-muted-foreground hover:text-foreground transition-colors">Features</Link>
              <Link to="#" className="text-muted-foreground hover:text-foreground transition-colors">Use Cases</Link>
              <Link to="#" className="text-muted-foreground hover:text-foreground transition-colors">Pricing</Link>
              <Link to="#" className="text-muted-foreground hover:text-foreground transition-colors">API</Link>
              <Link to="#" className="text-muted-foreground hover:text-foreground transition-colors">Integration</Link>
            </nav>
          </div>
          
          <div>
            <h3 className="font-medium text-lg mb-4">Company</h3>
            <nav className="flex flex-col space-y-3">
              <Link to="#" className="text-muted-foreground hover:text-foreground transition-colors">About</Link>
              <Link to="#" className="text-muted-foreground hover:text-foreground transition-colors">Blog</Link>
              <Link to="#" className="text-muted-foreground hover:text-foreground transition-colors">Careers</Link>
              <Link to="#" className="text-muted-foreground hover:text-foreground transition-colors">Contact</Link>
              <Link to="#" className="text-muted-foreground hover:text-foreground transition-colors">Partners</Link>
            </nav>
          </div>
        </div>
        
        <div className="border-t mt-12 pt-6 flex flex-col sm:flex-row justify-between items-center">
          <p className="text-sm text-muted-foreground">
            &copy; 2025 GeoScopeAI. All rights reserved.
          </p>
          <div className="flex space-x-6 mt-4 sm:mt-0">
            <Link to="#" className="text-sm text-muted-foreground hover:text-foreground">Privacy Policy</Link>
            <Link to="#" className="text-sm text-muted-foreground hover:text-foreground">Terms of Service</Link>
            <Link to="#" className="text-sm text-muted-foreground hover:text-foreground">Cookie Settings</Link>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
