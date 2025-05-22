
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Anomaly } from "@/data/sampleData";

interface AnomalyCardProps {
  anomaly: Anomaly;
}

const AnomalyCard = ({ anomaly }: AnomalyCardProps) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "new":
        return "bg-destructive";
      case "investigating":
        return "bg-amber-500";
      case "resolved":
        return "bg-green-500";
      default:
        return "bg-gray-500";
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case "fire":
        return "bg-fire-100 text-fire-800 dark:bg-fire-900/20 dark:text-fire-300";
      case "flood":
        return "bg-water-100 text-water-800 dark:bg-water-900/20 dark:text-water-300";
      case "deforestation":
        return "bg-forest-100 text-forest-800 dark:bg-forest-900/20 dark:text-forest-300";
      case "erosion":
        return "bg-earth-100 text-earth-800 dark:bg-earth-900/20 dark:text-earth-300";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300";
    }
  };

  return (
    <Card className={`anomaly-highlight anomaly-${anomaly.type} overflow-hidden`}>
      <div className="relative">
        <img
          src={anomaly.imageUrl}
          alt={`${anomaly.type} anomaly`}
          className="w-full h-48 object-cover"
        />
        <div className="absolute top-3 right-3 flex gap-2">
          <Badge className={getTypeColor(anomaly.type)} variant="secondary">
            {anomaly.type.charAt(0).toUpperCase() + anomaly.type.slice(1)}
          </Badge>
          <span 
            className={`${getStatusColor(anomaly.status)} px-2 py-1 rounded-md text-xs text-white font-medium`}
          >
            {anomaly.status.charAt(0).toUpperCase() + anomaly.status.slice(1)}
          </span>
        </div>
      </div>

      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <h3 className="font-semibold text-lg">{anomaly.location.name}</h3>
          <div className="text-sm text-muted-foreground">
            {new Date(anomaly.detectedAt).toLocaleDateString()}
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <div className="space-y-3">
          <div className="text-sm text-muted-foreground line-clamp-2">
            {anomaly.description}
          </div>
          <div className="flex items-center justify-between">
            <div className="text-sm">
              <span className="font-medium">Confidence:</span> {(anomaly.confidence * 100).toFixed(1)}%
            </div>
            <div className="text-sm">
              <span className="font-medium">Lat/Lng:</span> {anomaly.location.lat.toFixed(2)}, {anomaly.location.lng.toFixed(2)}
            </div>
          </div>
        </div>
      </CardContent>

      <CardFooter>
        <div className="flex justify-between items-center w-full">
          <Button variant="outline" size="sm">View Details</Button>
          <Button size="sm">Investigate</Button>
        </div>
      </CardFooter>
    </Card>
  );
};

export default AnomalyCard;
