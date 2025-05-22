
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { statsData } from "@/data/sampleData";
import AnomalyCard from "./AnomalyCard";
import MapView from "./MapView";
import { anomalies } from "@/data/sampleData";
import { useEffect, useState } from "react";

const Dashboard = () => {
  const [progressValue, setProgressValue] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setProgressValue(100);
    }, 500);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    <section id="dashboard" className="py-12 md:py-24 bg-muted/30">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col gap-4 md:gap-8">
          <div className="space-y-2">
            <h2 className="text-3xl font-bold tracking-tighter">Live Anomaly Dashboard</h2>
            <p className="text-muted-foreground">
              Monitor and analyze environmental anomalies detected by GeoScopeAI
            </p>
          </div>

          <Tabs defaultValue="map" className="w-full">
            <div className="flex justify-between items-center">
              <TabsList className="grid w-full max-w-md grid-cols-2">
                <TabsTrigger value="map">Map View</TabsTrigger>
                <TabsTrigger value="list">List View</TabsTrigger>
              </TabsList>
              
              <div className="hidden md:flex items-center gap-3 text-sm">
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-fire-600"></span>
                  Fire
                </span>
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-water-600"></span>
                  Flood
                </span>
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-forest-600"></span>
                  Deforestation
                </span>
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-earth-600"></span>
                  Erosion
                </span>
              </div>
            </div>
            
            <TabsContent value="map" className="mt-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card className="md:col-span-2">
                  <CardHeader className="pb-2">
                    <CardTitle>Global Anomaly Map</CardTitle>
                    <CardDescription>Interactive visualization of detected anomalies</CardDescription>
                  </CardHeader>
                  <CardContent className="p-0">
                    <div className="h-[500px] rounded-b-lg overflow-hidden">
                      <MapView anomalies={anomalies} />
                    </div>
                  </CardContent>
                </Card>
                
                <div className="flex flex-col gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Detection Summary</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <div className="flex justify-between mb-2 text-sm">
                          <span>Processing Speed</span>
                          <span>100%</span>
                        </div>
                        <Progress value={progressValue} />
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Total Anomalies</span>
                          <span className="font-medium">{statsData.totalAnomalies}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Active Alerts</span>
                          <span className="font-medium">{statsData.activeAlerts}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Resolved</span>
                          <span className="font-medium">{statsData.resolvedAnomalies}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle>Recent Activity</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {statsData.recentActivity.map((activity) => (
                          <div key={activity.id} className="flex flex-col space-y-1 text-sm">
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 bg-primary rounded-full"></div>
                              <span className="font-medium">{activity.event}</span>
                            </div>
                            <div className="text-xs text-muted-foreground ml-4">
                              {activity.location} • {new Date(activity.timestamp).toLocaleTimeString()}
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                    <CardFooter className="pt-0">
                      <div className="text-xs text-muted-foreground">
                        Last updated: {new Date(statsData.lastUpdated).toLocaleString()}
                      </div>
                    </CardFooter>
                  </Card>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="list" className="mt-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {anomalies.map((anomaly) => (
                  <AnomalyCard key={anomaly.id} anomaly={anomaly} />
                ))}
              </div>
            </TabsContent>
          </Tabs>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <Card className="bg-gradient-to-br from-fire-100 to-fire-50 dark:from-fire-900/20 dark:to-background">
              <CardHeader className="pb-2">
                <CardTitle className="text-fire-900 dark:text-fire-400 text-lg">Fires</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-fire-800 dark:text-fire-300">
                  {statsData.anomaliesByType.fire}
                </div>
                <p className="text-sm text-fire-700 dark:text-fire-400">Detected anomalies</p>
              </CardContent>
            </Card>
            
            <Card className="bg-gradient-to-br from-water-100 to-water-50 dark:from-water-900/20 dark:to-background">
              <CardHeader className="pb-2">
                <CardTitle className="text-water-900 dark:text-water-400 text-lg">Floods</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-water-800 dark:text-water-300">
                  {statsData.anomaliesByType.flood}
                </div>
                <p className="text-sm text-water-700 dark:text-water-400">Detected anomalies</p>
              </CardContent>
            </Card>
            
            <Card className="bg-gradient-to-br from-forest-100 to-forest-50 dark:from-forest-900/20 dark:to-background">
              <CardHeader className="pb-2">
                <CardTitle className="text-forest-900 dark:text-forest-400 text-lg">Deforestation</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-forest-800 dark:text-forest-300">
                  {statsData.anomaliesByType.deforestation}
                </div>
                <p className="text-sm text-forest-700 dark:text-forest-400">Detected anomalies</p>
              </CardContent>
            </Card>
            
            <Card className="bg-gradient-to-br from-earth-100 to-earth-50 dark:from-earth-900/20 dark:to-background">
              <CardHeader className="pb-2">
                <CardTitle className="text-earth-900 dark:text-earth-400 text-lg">Erosion</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-earth-800 dark:text-earth-300">
                  {statsData.anomaliesByType.erosion}
                </div>
                <p className="text-sm text-earth-700 dark:text-earth-400">Detected anomalies</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Dashboard;
