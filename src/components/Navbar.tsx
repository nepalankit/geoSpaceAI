
import { useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { scrollToSection } from "@/utils/scrollUtils";

const Navbar = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const handleNavClick = (sectionId: string, event: React.MouseEvent) => {
    event.preventDefault();
    scrollToSection(sectionId);
    setIsMobileMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-50 w-full bg-background/90 backdrop-blur-sm border-b">
      <div className="container flex items-center justify-between h-16 px-4 md:px-6">
        <Link to="/" className="flex items-center gap-2">
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

        {/* Desktop menu */}
        <nav className="hidden md:flex items-center gap-6">
          <a 
            href="#home" 
            className="text-sm font-medium hover:text-primary transition-colors"
            onClick={(e) => handleNavClick('home', e)}
          >
            Home
          </a>
          <a 
            href="#dashboard" 
            className="text-sm font-medium hover:text-primary transition-colors"
            onClick={(e) => handleNavClick('dashboard', e)}
          >
            Dashboard
          </a>
          <a 
            href="#features" 
            className="text-sm font-medium hover:text-primary transition-colors"
            onClick={(e) => handleNavClick('features', e)}
          >
            Features
          </a>
          <a 
            href="#about" 
            className="text-sm font-medium hover:text-primary transition-colors"
            onClick={(e) => handleNavClick('about', e)}
          >
            About
          </a>
        </nav>

        <div className="hidden md:flex items-center gap-4">
          <Button variant="outline">Log In</Button>
          <Button>Try Demo</Button>
        </div>

        {/* Mobile menu button */}
        <button
          className="md:hidden p-2 rounded-md"
          onClick={toggleMobileMenu}
          aria-label="Toggle menu"
        >
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
            className="h-6 w-6"
          >
            <line x1="4" x2="20" y1="12" y2="12" />
            <line x1="4" x2="20" y1="6" y2="6" />
            <line x1="4" x2="20" y1="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Mobile menu */}
      <div className={cn(
        "fixed inset-x-0 top-16 z-50 bg-background border-b md:hidden",
        isMobileMenuOpen ? "block" : "hidden"
      )}>
        <div className="container px-4 py-4 flex flex-col space-y-4">
          <a 
            href="#home" 
            className="text-sm font-medium hover:text-primary transition-colors py-2"
            onClick={(e) => handleNavClick('home', e)}
          >
            Home
          </a>
          <a 
            href="#dashboard" 
            className="text-sm font-medium hover:text-primary transition-colors py-2"
            onClick={(e) => handleNavClick('dashboard', e)}
          >
            Dashboard
          </a>
          <a 
            href="#features" 
            className="text-sm font-medium hover:text-primary transition-colors py-2"
            onClick={(e) => handleNavClick('features', e)}
          >
            Features
          </a>
          <a 
            href="#about" 
            className="text-sm font-medium hover:text-primary transition-colors py-2"
            onClick={(e) => handleNavClick('about', e)}
          >
            About
          </a>
          <div className="flex flex-col space-y-2 pt-2">
            <Button variant="outline">Log In</Button>
            <Button>Try Demo</Button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
