
export type AnomalyType = 'fire' | 'flood' | 'deforestation' | 'erosion';

export interface Anomaly {
  id: string;
  type: AnomalyType;
  location: {
    lat: number;
    lng: number;
    name: string;
  };
  detectedAt: string;
  confidence: number;
  imageUrl: string;
  status: 'new' | 'investigating' | 'resolved';
  description: string;
}

export const anomalies: Anomaly[] = [
  {
    id: '1',
    type: 'fire',
    location: {
      lat: 34.05,
      lng: -118.25,
      name: 'Los Angeles, CA'
    },
    detectedAt: '2023-08-15T08:30:00Z',
    confidence: 0.92,
    imageUrl: 'https://images.unsplash.com/photo-1433086966358-54859d0ed716',
    status: 'new',
    description: 'Possible wildfire detected in mountainous region. High temperature anomaly with smoke signature detected by thermal imaging.'
  },
  {
    id: '2',
    type: 'flood',
    location: {
      lat: 29.76,
      lng: -95.37,
      name: 'Houston, TX'
    },
    detectedAt: '2023-09-10T14:15:00Z',
    confidence: 0.88,
    imageUrl: 'https://images.unsplash.com/photo-1482938289607-e9573fc25ebb',
    status: 'investigating',
    description: 'Flood anomaly detected in residential area. Significant increase in water level and surface coverage compared to baseline.'
  },
  {
    id: '3',
    type: 'deforestation',
    location: {
      lat: -3.47,
      lng: -62.21,
      name: 'Amazon Rainforest, Brazil'
    },
    detectedAt: '2023-07-28T10:45:00Z',
    confidence: 0.95,
    imageUrl: 'https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9',
    status: 'investigating',
    description: 'Illegal deforestation activity detected. Pattern of tree removal inconsistent with authorized logging operations.'
  },
  {
    id: '4',
    type: 'erosion',
    location: {
      lat: 38.62,
      lng: -90.19,
      name: 'St. Louis, MO'
    },
    detectedAt: '2023-06-20T16:00:00Z',
    confidence: 0.79,
    imageUrl: 'https://images.unsplash.com/photo-1513836279014-a89f7a76ae86',
    status: 'resolved',
    description: 'Land degradation and erosion detected along riverbank. Progressive loss of vegetation and soil stability observed.'
  },
  {
    id: '5',
    type: 'fire',
    location: {
      lat: 37.77,
      lng: -122.42,
      name: 'San Francisco, CA'
    },
    detectedAt: '2023-10-05T09:20:00Z',
    confidence: 0.91,
    imageUrl: 'https://images.unsplash.com/photo-1469474968028-56623f02e42e',
    status: 'new',
    description: 'Thermal anomaly detected in urban-wildland interface. Pattern matches early-stage wildfire development.'
  },
  {
    id: '6',
    type: 'flood',
    location: {
      lat: 27.95,
      lng: 82.46,
      name: 'Tampa, FL'
    },
    detectedAt: '2023-09-25T13:10:00Z',
    confidence: 0.85,
    imageUrl: 'https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05',
    status: 'new',
    description: 'Coastal flooding detected following tropical storm. Anomalous water coverage detected in previously dry urban areas.'
  }
];

export const statsData = {
  totalAnomalies: 342,
  resolvedAnomalies: 278,
  activeAlerts: 64,
  lastUpdated: '2023-10-10T12:00:00Z',
  anomaliesByType: {
    fire: 112,
    flood: 87,
    deforestation: 95,
    erosion: 48
  },
  recentActivity: [
    {
      id: '1001',
      event: 'New fire detected',
      location: 'Sierra Nevada, CA',
      timestamp: '2023-10-10T11:45:00Z'
    },
    {
      id: '1002',
      event: 'Deforestation alert verified',
      location: 'Borneo, Indonesia',
      timestamp: '2023-10-10T10:30:00Z'
    },
    {
      id: '1003',
      event: 'Flood alert resolved',
      location: 'Mississippi Delta, LA',
      timestamp: '2023-10-10T09:15:00Z'
    }
  ]
};

export const featuresData = [
  {
    title: 'Wildfire Detection',
    description: 'Early detection of wildfires using thermal imaging and smoke analysis algorithms.',
    icon: 'flame',
    color: 'fire'
  },
  {
    title: 'Flood Monitoring',
    description: 'Identifying flood events by analyzing water body expansion and changes in surface reflectance.',
    icon: 'drop',
    color: 'water'
  },
  {
    title: 'Deforestation Tracking',
    description: 'Monitoring forest cover changes and detecting illegal logging activities.',
    icon: 'tree',
    color: 'forest'
  },
  {
    title: 'Land Degradation Analysis',
    description: 'Identifying erosion, soil degradation, and land use changes affecting ecosystem health.',
    icon: 'mountain',
    color: 'earth'
  }
];
