
import { useEffect, useRef } from "react";
import { Anomaly } from "@/data/sampleData";

interface MapViewProps {
  anomalies: Anomaly[];
}

const MapView = ({ anomalies }: MapViewProps) => {
  const mapRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (!mapRef.current) return;
    
    // For demonstration purposes - this is a placeholder map
    // In a real application, you'd implement an actual map library like Mapbox, Leaflet, or Google Maps
    
    const renderPlaceholderMap = () => {
      if (!mapRef.current) return;
      
      const canvas = document.createElement('canvas');
      canvas.width = mapRef.current.clientWidth;
      canvas.height = mapRef.current.clientHeight;
      mapRef.current.appendChild(canvas);
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      // Draw map background
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
      gradient.addColorStop(0, '#dde6e8');
      gradient.addColorStop(1, '#cad9db');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw grid lines
      ctx.strokeStyle = '#b1c5c9';
      ctx.lineWidth = 0.5;
      
      for (let i = 0; i < canvas.width; i += 40) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, canvas.height);
        ctx.stroke();
      }
      
      for (let i = 0; i < canvas.height; i += 40) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(canvas.width, i);
        ctx.stroke();
      }
      
      // Draw coast lines (simplified)
      ctx.strokeStyle = '#75a0a8';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(canvas.width * 0.1, canvas.height * 0.3);
      ctx.bezierCurveTo(
        canvas.width * 0.2, canvas.height * 0.2,
        canvas.width * 0.3, canvas.height * 0.4,
        canvas.width * 0.4, canvas.height * 0.3
      );
      ctx.bezierCurveTo(
        canvas.width * 0.5, canvas.height * 0.2,
        canvas.width * 0.6, canvas.height * 0.3,
        canvas.width * 0.7, canvas.height * 0.4
      );
      ctx.stroke();
      
      // Draw some land masses
      ctx.fillStyle = '#c8d8c8';
      
      // Land mass 1
      ctx.beginPath();
      ctx.moveTo(canvas.width * 0.15, 0);
      ctx.lineTo(canvas.width * 0.35, 0);
      ctx.lineTo(canvas.width * 0.4, canvas.height * 0.25);
      ctx.lineTo(canvas.width * 0.3, canvas.height * 0.35);
      ctx.lineTo(canvas.width * 0.2, canvas.height * 0.3);
      ctx.lineTo(canvas.width * 0.1, canvas.height * 0.15);
      ctx.closePath();
      ctx.fill();
      
      // Land mass 2
      ctx.beginPath();
      ctx.moveTo(canvas.width * 0.65, canvas.height * 0.4);
      ctx.lineTo(canvas.width * 0.8, canvas.height * 0.3);
      ctx.lineTo(canvas.width * 0.95, canvas.height * 0.4);
      ctx.lineTo(canvas.width, canvas.height * 0.5);
      ctx.lineTo(canvas.width, canvas.height * 0.7);
      ctx.lineTo(canvas.width * 0.8, canvas.height * 0.8);
      ctx.lineTo(canvas.width * 0.7, canvas.height * 0.6);
      ctx.closePath();
      ctx.fill();
      
      // Land mass 3
      ctx.beginPath();
      ctx.moveTo(0, canvas.height * 0.6);
      ctx.lineTo(canvas.width * 0.2, canvas.height * 0.5);
      ctx.lineTo(canvas.width * 0.3, canvas.height * 0.7);
      ctx.lineTo(canvas.width * 0.2, canvas.height * 0.9);
      ctx.lineTo(0, canvas.height * 0.8);
      ctx.closePath();
      ctx.fill();
      
      // Add anomaly markers
      anomalies.forEach((anomaly) => {
        // Convert lat/lng to canvas coordinates (simplified)
        const x = ((anomaly.location.lng + 180) / 360) * canvas.width;
        const y = ((90 - anomaly.location.lat) / 180) * canvas.height;
        
        // Get color based on anomaly type
        let color = '#000';
        switch(anomaly.type) {
          case 'fire': color = '#f44336'; break;
          case 'flood': color = '#2196f3'; break;
          case 'deforestation': color = '#4caf50'; break;
          case 'erosion': color = '#ff9800'; break;
        }
        
        // Draw marker
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw pulse animation effect
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.stroke();
      });
      
      // Add legend
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.fillRect(canvas.width - 130, canvas.height - 100, 120, 90);
      ctx.strokeStyle = '#999';
      ctx.strokeRect(canvas.width - 130, canvas.height - 100, 120, 90);
      
      ctx.font = '12px Arial';
      ctx.fillStyle = '#333';
      ctx.fillText('Anomaly Types', canvas.width - 120, canvas.height - 80);
      
      // Legend items
      const legendItems = [
        { color: '#f44336', label: 'Fire' },
        { color: '#2196f3', label: 'Flood' },
        { color: '#4caf50', label: 'Deforestation' },
        { color: '#ff9800', label: 'Erosion' }
      ];
      
      legendItems.forEach((item, i) => {
        const y = canvas.height - 65 + i * 15;
        ctx.fillStyle = item.color;
        ctx.beginPath();
        ctx.arc(canvas.width - 115, y, 4, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.fillStyle = '#333';
        ctx.fillText(item.label, canvas.width - 105, y + 4);
      });
    };
    
    renderPlaceholderMap();
    
    return () => {
      if (mapRef.current) {
        while (mapRef.current.firstChild) {
          mapRef.current.removeChild(mapRef.current.firstChild);
        }
      }
    };
  }, [anomalies]);
  
  return (
    <div className="relative w-full h-full">
      <div ref={mapRef} className="w-full h-full">
        {/* Map will be rendered here */}
      </div>
      <div className="absolute top-4 left-4 bg-background/80 backdrop-blur-sm p-3 rounded-md shadow text-sm">
        <div className="font-medium mb-1">Map Controls</div>
        <div className="grid grid-cols-2 gap-2">
          <button className="px-2 py-1 bg-secondary/20 rounded hover:bg-secondary/30">Zoom In</button>
          <button className="px-2 py-1 bg-secondary/20 rounded hover:bg-secondary/30">Zoom Out</button>
          <button className="px-2 py-1 bg-secondary/20 rounded hover:bg-secondary/30">Reset</button>
          <button className="px-2 py-1 bg-secondary/20 rounded hover:bg-secondary/30">Filters</button>
        </div>
      </div>
      <div className="absolute bottom-4 right-4 bg-background/80 backdrop-blur-sm px-3 py-1 rounded-md text-xs">
        This is a demo visualization. Connect to a real mapping API for production use.
      </div>
    </div>
  );
};

export default MapView;
