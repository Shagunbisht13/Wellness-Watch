import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  FileImage,
  Frown,
  Meh,
  Smile,
  Volume2,
  MessageSquare,
  ShieldCheck,
  Activity,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
} from 'recharts';
import { getLatestPrediction, getHistory, getStats, getProofUrl } from './api';
import './App.css';

const COLORS = {
  harassment: '#c45c4a',
  stress: '#c4923a',
  'normal behaviour': '#3d7a6a',
};

function statusTone(value) {
  if (!value) return 'stable';
  const v = String(value).toLowerCase();
  if (v.includes('harassment') || v.includes('angry') || v.includes('abusive')) {
    return 'alert';
  }
  if (v.includes('stress') || v.includes('sad')) return 'caution';
  return 'stable';
}

function FaceIcon({ emotion }) {
  const e = (emotion || '').toLowerCase();
  if (e.includes('angry') || e.includes('sad')) return <Frown size={16} />;
  if (e.includes('happy')) return <Smile size={16} />;
  return <Meh size={16} />;
}

function resultLabel(result) {
  const r = (result || '').toLowerCase();
  if (r === 'harassment') return 'Harassment';
  if (r === 'stress') return 'Stress';
  if (r === 'normal behaviour') return 'Stable';
  return result || '—';
}

function App() {
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({});

  useEffect(() => {
    fetchData();
    const poll = setInterval(fetchData, 3000);
    return () => clearInterval(poll);
  }, []);

  const fetchData = async () => {
    const [latestData, historyData, statsData] = await Promise.all([
      getLatestPrediction(),
      getHistory(),
      getStats(),
    ]);
    if (latestData && latestData.status !== 'no_data') setLatest(latestData);
    setHistory(Array.isArray(historyData) ? historyData : []);
    setStats(statsData || {});
  };

  const pieData = useMemo(
    () =>
      Object.keys(stats).map((key) => ({
        name: key,
        value: stats[key],
      })),
    [stats]
  );

  const totalIncidents = pieData.reduce((sum, d) => sum + d.value, 0);
  const proofRecords = history.filter((r) => r.proof_image || r.proof_audio);
  const verdict = (latest?.final_result || '').toLowerCase();
  const isHarassment = verdict === 'harassment';
  const isStress = verdict === 'stress';

  const statusTitle = isHarassment
    ? 'Harassment detected'
    : isStress
      ? 'Elevated stress indicators'
      : latest
        ? 'Environment appears stable'
        : 'Waiting for live analysis';

  const statusDetail = isHarassment
    ? 'Evidence has been retained. Review the case log and proof panel.'
    : isStress
      ? 'One or more signals are elevated. Continue monitoring.'
      : latest
        ? 'Face, voice, and language signals are within expected range.'
        : 'Start the detection pipeline to begin live updates.';

  return (
    <div className="ww">
      <header className="ww-top">
        <div className="ww-brand">
          <span className="ww-mark" aria-hidden />
          <div>
            <h1>Wellness Watch</h1>
            <p>Workplace wellbeing &amp; behaviour monitoring</p>
          </div>
        </div>
        <div className="ww-top-meta">
          <span className={`ww-live ${latest ? 'on' : ''}`}>
            <span className="ww-dot" />
            {latest ? 'Live' : 'Offline'}
          </span>
          <span className="ww-time">
            {latest?.timestamp
              ? `Updated ${String(latest.timestamp).split(' ').pop()}`
              : 'No recent update'}
          </span>
        </div>
      </header>

      <section
        className={`ww-status ${
          isHarassment ? 'alert' : isStress ? 'caution' : 'stable'
        }`}
      >
        <div className="ww-status-icon">
          {isHarassment || isStress ? (
            <AlertTriangle size={22} />
          ) : (
            <CheckCircle2 size={22} />
          )}
        </div>
        <div className="ww-status-copy">
          <h2>{statusTitle}</h2>
          <p>{statusDetail}</p>
        </div>
        {isHarassment && (
          <a className="ww-status-cta" href="#evidence">
            View evidence
          </a>
        )}
      </section>

      <section className="ww-metrics" aria-label="Live signals">
        <article className="ww-metric">
          <header>
            <span>Face emotion</span>
            <Activity size={16} />
          </header>
          <strong className={`tone-${statusTone(latest?.face_emotion)}`}>
            {latest?.face_emotion || '—'}
          </strong>
        </article>
        <article className="ww-metric">
          <header>
            <span>Voice tone</span>
            <Volume2 size={16} />
          </header>
          <strong className={`tone-${statusTone(latest?.speech_emotion)}`}>
            {latest?.speech_emotion || '—'}
          </strong>
        </article>
        <article className="ww-metric">
          <header>
            <span>Language class</span>
            <MessageSquare size={16} />
          </header>
          <strong className={`tone-${statusTone(latest?.text_prediction)}`}>
            {latest?.text_prediction || '—'}
          </strong>
        </article>
        <article className="ww-metric emphasis">
          <header>
            <span>Overall result</span>
            <ShieldCheck size={16} />
          </header>
          <strong className={`tone-${statusTone(latest?.final_result)}`}>
            {resultLabel(latest?.final_result)}
          </strong>
        </article>
      </section>

      <div className="ww-grid">
        <section className="ww-panel" aria-labelledby="log-heading">
          <div className="ww-panel-head">
            <h3 id="log-heading">Activity log</h3>
            <span>{history.length} recent events</span>
          </div>
          <div className="ww-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Face</th>
                  <th>Voice</th>
                  <th>Language</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {history.length === 0 && (
                  <tr>
                    <td colSpan={5} className="empty">
                      No events yet. Run the detection pipeline to populate this
                      log.
                    </td>
                  </tr>
                )}
                {history.slice(0, 18).map((row) => {
                  const tone = statusTone(row.final_result);
                  return (
                    <tr
                      key={row.id || row._id || row.timestamp}
                      className={tone === 'alert' ? 'row-alert' : ''}
                    >
                      <td className="mono">
                        {String(row.timestamp || '').split(' ').pop()}
                      </td>
                      <td>
                        <span className={`inline tone-${statusTone(row.face_emotion)}`}>
                          <FaceIcon emotion={row.face_emotion} />
                          {row.face_emotion}
                        </span>
                      </td>
                      <td>{row.speech_emotion}</td>
                      <td>{row.text_prediction}</td>
                      <td>
                        <span className={`badge ${tone}`}>
                          {resultLabel(row.final_result)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <aside className="ww-side">
          <section className="ww-panel" aria-labelledby="dist-heading">
            <div className="ww-panel-head">
              <h3 id="dist-heading">Outcome mix</h3>
            </div>
            <div className="ww-chart">
              <div className="ww-donut">
                <ResponsiveContainer width="100%" height={160}>
                  <PieChart>
                    <Pie
                      data={
                        pieData.length
                          ? pieData
                          : [{ name: 'idle', value: 1 }]
                      }
                      dataKey="value"
                      innerRadius={48}
                      outerRadius={68}
                      paddingAngle={2}
                      stroke="none"
                    >
                      {(pieData.length
                        ? pieData
                        : [{ name: 'idle', value: 1 }]
                      ).map((entry) => (
                        <Cell
                          key={entry.name}
                          fill={
                            COLORS[entry.name] ||
                            (entry.name === 'idle' ? '#d7ddd9' : '#8a938e')
                          }
                        />
                      ))}
                    </Pie>
                    <RechartsTooltip
                      contentStyle={{
                        background: '#fff',
                        border: '1px solid #d5ddd8',
                        borderRadius: 8,
                        fontFamily: 'Outfit, sans-serif',
                        fontSize: 13,
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="ww-donut-center">
                  <strong>{totalIncidents || 0}</strong>
                  <span>total</span>
                </div>
              </div>
              <ul className="ww-legend">
                {pieData.map((d) => (
                  <li key={d.name}>
                    <i style={{ background: COLORS[d.name] || '#8a938e' }} />
                    <span>
                      {resultLabel(d.name)}
                      <em>
                        {totalIncidents
                          ? ` · ${Math.round((d.value / totalIncidents) * 100)}%`
                          : ''}
                      </em>
                    </span>
                  </li>
                ))}
                {!pieData.length && (
                  <li className="muted">No outcomes recorded yet</li>
                )}
              </ul>
            </div>
          </section>

          <section
            className="ww-panel"
            id="evidence"
            aria-labelledby="evidence-heading"
          >
            <div className="ww-panel-head">
              <h3 id="evidence-heading">Evidence</h3>
              <span>Saved on harassment only</span>
            </div>
            {proofRecords.length === 0 ? (
              <p className="ww-empty">
                No evidence files yet. Clips and face stills appear here when
                harassment is confirmed.
              </p>
            ) : (
              <div className="ww-evidence-list">
                {proofRecords.slice(0, 3).map((proof) => (
                  <article
                    key={proof.id || proof._id}
                    className="ww-evidence"
                  >
                    <div className="ww-evidence-media">
                      {proof.proof_image ? (
                        <img
                          src={getProofUrl(proof.proof_image)}
                          alt="Captured face evidence"
                        />
                      ) : (
                        <div className="ww-evidence-ph">
                          <FileImage size={20} />
                        </div>
                      )}
                    </div>
                    <div className="ww-evidence-meta">
                      <time>{proof.timestamp}</time>
                      <p>
                        Face <strong>{proof.face_emotion}</strong>
                        {' · '}
                        Language <strong>{proof.text_prediction}</strong>
                      </p>
                      {proof.proof_audio && (
                        <audio
                          controls
                          src={getProofUrl(proof.proof_audio)}
                          className="ww-audio"
                        />
                      )}
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        </aside>
      </div>

      <footer className="ww-foot">
        <span>Wellness Watch · Multimodal behaviour analysis</span>
        <span>Modules 1–13 connected via live API</span>
      </footer>
    </div>
  );
}

export default App;
