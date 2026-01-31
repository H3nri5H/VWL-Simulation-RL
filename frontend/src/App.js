import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

function App() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory();
    const interval = setInterval(fetchHistory, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/training/history');
      const data = await response.json();
      setHistory(data.iterations || []);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching history:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="App">
        <div className="loading">Loading training data...</div>
      </div>
    );
  }

  const latest = history.length > 0 ? history[history.length - 1] : null;

  return (
    <div className="App">
      <header>
        <h1>VWL Simulation Training Dashboard</h1>
      </header>

      {latest && (
        <div className="stats">
          <div className="stat-card">
            <div className="stat-label">Iteration</div>
            <div className="stat-value">{latest.iteration}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Reward Mean</div>
            <div className="stat-value">{latest.reward_mean.toFixed(2)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Episode Length</div>
            <div className="stat-value">{latest.episode_len.toFixed(0)}</div>
          </div>
        </div>
      )}

      <div className="charts">
        <div className="chart-container">
          <h2>Episode Reward</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1f3a" />
              <XAxis dataKey="iteration" stroke="#8884d8" />
              <YAxis stroke="#8884d8" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1a1f3a', border: 'none' }}
                labelStyle={{ color: '#8884d8' }}
              />
              <Legend />
              <Line type="monotone" dataKey="reward_mean" stroke="#82ca9d" strokeWidth={2} dot={false} name="Mean" />
              <Line type="monotone" dataKey="reward_max" stroke="#8884d8" strokeWidth={1} dot={false} name="Max" />
              <Line type="monotone" dataKey="reward_min" stroke="#ffc658" strokeWidth={1} dot={false} name="Min" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h2>Episode Length</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1f3a" />
              <XAxis dataKey="iteration" stroke="#8884d8" />
              <YAxis stroke="#8884d8" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1a1f3a', border: 'none' }}
                labelStyle={{ color: '#8884d8' }}
              />
              <Legend />
              <Line type="monotone" dataKey="episode_len" stroke="#82ca9d" strokeWidth={2} dot={false} name="Length" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default App;
