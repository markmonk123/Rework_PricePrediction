```javascript
import React from 'react';

const RealTimeData = ({ marketData }) => {
  if (!marketData) return <div>Loading real-time market data...</div>;

  return (
    <div style={{ padding: '20px' }}>
      <h2>Real-Time Market Data</h2>
      <p><strong>Latest Price:</strong> ${marketData.latest_price}</p>
      <p><strong>Timestamp:</strong> {marketData.time}</p>
    </div>
  );
};

export default RealTimeData;
```