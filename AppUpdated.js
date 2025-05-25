```javascript
import React, { useEffect, useState } from 'react';
import RealTimeData from './RealTimeData';
import HistoricalData from './HistoricalData';
import FuturePredictions from './FuturePredictions';

const App = () => {
  const [marketData, setMarketData] = useState(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8080'); // Connect to WebSocket server

    ws.onmessage = (message) => {
      const data = JSON.parse(message.data);
      setMarketData(data); // Update state with received data
    };

    ws.onclose = () => console.log('WebSocket disconnected');
    ws.onerror = (error) => console.error('WebSocket error:', error);

    return () => ws.close(); // Cleanup connection on component unmount
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <div style={{ flex: 1, borderBottom: '1px solid black' }}>
        <RealTimeData marketData={marketData} />
      </div>
      <div style={{ flex: 1, display: 'flex' }}>
        <div style={{ flex: 1, borderRight: '1px solid black' }}>
          <HistoricalData marketData={marketData} />
        </div>
        <div style={{ flex: 1 }}>
          <FuturePredictions marketData={marketData} />
        </div>
      </div>
    </div>
  );
};

export default App;
```