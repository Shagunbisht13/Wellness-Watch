import React, { useState, useEffect } from 'react';
import { getLatestPrediction, getHistory, getStats, getProofUrl } from './api';
import { AlertCircle, Activity, ShieldCheck, Clock, Image as ImageIcon, Volume2 } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, Legend } from 'recharts';
import './App.css';

const COLORS = {
  'harassment': '#ef4444',
  'stress': '#f97316',
  'normal behaviour': '#10b981',
};

function App() {
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({});

  useEffect(() => {
    // Initial fetch
    fetchData();

    // Poll every 3 seconds for live updates
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    const [latestData, historyData, statsData] = await Promise.all([
      getLatestPrediction(),
      getHistory(),
      getStats()
    ]);
    
    if (latestData && latestData.status !== "no_data") {
      setLatest(latestData);
    }
    setHistory(historyData);
    setStats(statsData);
  };

  const getStatusClass = (value, type = 'text') => {
    if (!value) return '';
    const v = value.toLowerCase();
    if (v.includes('harassment') || v.includes('angry') || v.includes('abusive')) return `status-harassment`;
    if (v.includes('stress') || v.includes('sad')) return `status-stress`;
    return `status-normal`;
  };

  // Prepare chart data
  const pieData = Object.keys(stats).map(key => ({
    name: key,
    value: stats[key]
  }));

  // Filter out records that have proof
  const proofRecords = history.filter(r => r.proof_image || r.proof_audio);

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>🛡️ Harassment Detection System</h1>
        <p>Real-time multimodal behavioral analysis</p>
      </header>

      {/* Alert Banner */}
      {latest && latest.final_result === 'harassment' && (
        <div className="alert-banner alert-harassment glass-panel">
          <AlertCircle size={32} />
          <div>
            <div>HARASSMENT DETECTED</div>
            <div style={{ fontSize: '0.9rem', fontWeight: 'normal', opacity: 0.8 }}>Immediate attention required — Evidence saved</div>
          </div>
        </div>
      )}
      {latest && latest.final_result === 'stress' && (
        <div className="alert-banner alert-stress glass-panel">
          <Activity size={32} />
          <div>
            <div>STRESS BEHAVIOUR DETECTED</div>
            <div style={{ fontSize: '0.9rem', fontWeight: 'normal', opacity: 0.8 }}>Monitoring closely</div>
          </div>
        </div>
      )}
      {latest && latest.final_result === 'normal behaviour' && (
        <div className="alert-banner alert-normal glass-panel">
          <ShieldCheck size={32} />
          <div>
            <div>NORMAL BEHAVIOUR</div>
            <div style={{ fontSize: '0.9rem', fontWeight: 'normal', opacity: 0.8 }}>All signals within normal range</div>
          </div>
        </div>
      )}

      {/* Live Metrics Grid */}
      <div className="metrics-grid">
        <div className="metric-card glass-panel">
          <div className="metric-label"><Clock size={16} style={{display:'inline', verticalAlign:'middle', marginRight:'5px'}}/> Last Updated</div>
          <div className="metric-value" style={{fontSize: '1.2rem', color: 'var(--text-muted)'}}>
            {latest ? latest.timestamp.split(' ')[1] : '—'}
          </div>
        </div>
        <div className="metric-card glass-panel">
          <div className="metric-label">😐 Face Emotion</div>
          <div className={`metric-value ${getStatusClass(latest?.face_emotion)}`}>
            {latest ? latest.face_emotion.toUpperCase() : '—'}
          </div>
        </div>
        <div className="metric-card glass-panel">
          <div className="metric-label">🔊 Voice Tone</div>
          <div className={`metric-value ${getStatusClass(latest?.speech_emotion)}`}>
            {latest ? latest.speech_emotion.toUpperCase() : '—'}
          </div>
        </div>
        <div className="metric-card glass-panel">
          <div className="metric-label">📝 Text Analysis</div>
          <div className={`metric-value ${getStatusClass(latest?.text_prediction)}`}>
            {latest ? latest.text_prediction.toUpperCase() : '—'}
          </div>
        </div>
        <div className="metric-card glass-panel" style={{border: '1px solid rgba(255,255,255,0.2)'}}>
          <div className="metric-label">⚖️ Final Result</div>
          <div className={`metric-value ${getStatusClass(latest?.final_result)}`}>
            {latest ? latest.final_result.toUpperCase() : '—'}
          </div>
        </div>
      </div>

      <div className="main-content">
        {/* Left Column: History Table */}
        <div className="history-section">
          <div className="glass-panel">
            <h2 className="section-title">📋 Prediction Log</h2>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Face</th>
                    <th>Voice</th>
                    <th>Text</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice(0, 15).map((row) => {
                    let badgeClass = 'badge-normal';
                    if (row.final_result === 'harassment') badgeClass = 'badge-harassment';
                    if (row.final_result === 'stress') badgeClass = 'badge-stress';
                    
                    return (
                      <tr key={row._id}>
                        <td>{row.timestamp.split(' ')[1]}</td>
                        <td>{row.face_emotion}</td>
                        <td>{row.speech_emotion}</td>
                        <td>{row.text_prediction}</td>
                        <td>
                          <span className={`badge ${badgeClass}`}>
                            {row.final_result}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right Column: Stats & Proof */}
        <div className="sidebar-section" style={{display: 'flex', flexDirection: 'column', gap: '2rem'}}>
          
          <div className="glass-panel">
            <h2 className="section-title">📈 Statistics</h2>
            <div style={{ height: 250 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[entry.name] || '#8884d8'} />
                    ))}
                  </Pie>
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend verticalAlign="bottom" height={36}/>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass-panel">
            <h2 className="section-title">🗂️ Proof Evidence</h2>
            {proofRecords.length === 0 ? (
              <p style={{color: 'var(--text-muted)'}}>No proof evidence stored yet. Proof is saved only when harassment is detected.</p>
            ) : (
              <div className="proof-grid">
                {proofRecords.slice(0, 3).map((proof) => (
                  <div key={proof._id} className="proof-card">
                    <div className="proof-header">
                      <span>{proof.timestamp}</span>
                      <span style={{color: 'var(--color-harassment)', fontWeight: 'bold'}}>HARASSMENT</span>
                    </div>
                    <div className="proof-media">
                      <div>
                        {proof.proof_image ? (
                           <img src={getProofUrl(proof.proof_image)} alt="Proof" className="proof-image" />
                        ) : (
                          <div style={{background:'rgba(0,0,0,0.3)', height:'100px', display:'flex', alignItems:'center', justifyContent:'center', borderRadius:'8px'}}>
                            <ImageIcon color="rgba(255,255,255,0.2)"/>
                          </div>
                        )}
                      </div>
                      <div className="proof-details">
                        <div style={{fontSize:'0.85rem'}}>
                          <div style={{marginBottom:'4px'}}><span style={{color:'var(--text-muted)'}}>Face:</span> {proof.face_emotion}</div>
                          <div style={{marginBottom:'4px'}}><span style={{color:'var(--text-muted)'}}>Text:</span> {proof.text_prediction}</div>
                        </div>
                        {proof.proof_audio && (
                          <div style={{marginTop: 'auto'}}>
                            <audio controls className="audio-player" src={getProofUrl(proof.proof_audio)}>
                              Your browser does not support the audio element.
                            </audio>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
