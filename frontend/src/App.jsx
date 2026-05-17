import React, { useEffect, useState } from 'react'

const COLORS = {
  bg: '#0d1117',
  green: '#00ff85',
  white: '#ffffff',
  yellow: '#ffd166',
  red: '#ff4d4f',
  card: '#0f1720',
  darkGreen: '#1a4d2e',
  darkBlue: '#1a3a52',
  darkRed: '#4d1f1f',
}

export default function App() {
  const [activeTab, setActiveTab] = useState('predictor')
  const [teams, setTeams] = useState([])
  const [loadingTeams, setLoadingTeams] = useState(true)
  const [teamsError, setTeamsError] = useState(null)
  const [home, setHome] = useState('')
  const [away, setAway] = useState('')
  const [predictLoading, setPredictLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [simulationData, setSimulationData] = useState(null)
  const [simulationLoading, setSimulationLoading] = useState(false)

  useEffect(() => {
    let mounted = true
    async function fetchTeams() {
      try {
        setLoadingTeams(true)
        const res = await fetch('/api/teams')
        if (!res.ok) throw new Error('Failed to fetch teams')
        const data = await res.json()
        if (mounted) setTeams(data.teams || [])
      } catch (err) {
        setTeamsError(err.message)
      } finally {
        setLoadingTeams(false)
      }
    }
    fetchTeams()
    return () => { mounted = false }
  }, [])

  async function predict() {
    setError(null)
    setResult(null)
    if (!home || !away) {
      setError('Please select both teams')
      return
    }
    if (home === away) {
      setError('Home and away teams must differ')
      return
    }
    try {
      setPredictLoading(true)
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home_team: home, away_team: away })
      })
      if (!res.ok) throw new Error('Prediction failed')
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setPredictLoading(false)
    }
  }

  async function runSimulation() {
    try {
      setSimulationLoading(true)
      const res = await fetch('/api/simulate')
      if (!res.ok) throw new Error('Simulation failed')
      const data = await res.json()
      setSimulationData(data.table)
    } catch (err) {
      setError(err.message)
    } finally {
      setSimulationLoading(false)
    }
  }

  const getRowColor = (position) => {
    if (position <= 4) return COLORS.darkGreen
    if (position <= 6) return COLORS.darkBlue
    if (position >= 18) return COLORS.darkRed
    return COLORS.card
  }

  const emptyState = (
    <div className="empty">Select two teams to get a prediction</div>
  )

  const bar = (label, pct, color) => (
    <div className="prob-row">
      <div className="prob-label">{label}</div>
      <div className="bar-outer">
        <div className="bar-inner" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="prob-num">{Math.round(pct)}%</div>
    </div>
  )

  return (
    <div className="app" style={{ background: COLORS.bg, minHeight: '100vh', color: COLORS.white, fontFamily: 'Inter, system-ui, sans-serif' }}>
      <style>{`
        .container{max-width:900px;margin:0 auto;padding:48px 16px;text-align:center}
        .header{font-weight:900;font-size:28px;display:flex;align-items:center;justify-content:center;gap:12px}
        .accent{color:${COLORS.green}}
        .subtitle{color:#b8c2cc;margin-top:6px}
        .tab-switcher{display:flex;gap:8px;justify-content:center;margin-bottom:32px;border-bottom:1px solid #1f2937}
        .tab-btn{padding:12px 24px;border:none;background:transparent;color:#b8c2cc;cursor:pointer;font-weight:600;border-bottom:3px solid transparent}
        .tab-btn.active{color:${COLORS.green};border-bottom-color:${COLORS.green}}
        .selector{display:flex;align-items:center;justify-content:center;gap:12px;margin:28px 0}
        select{background:#0b0f13;color:${COLORS.white};border:1px solid transparent;padding:10px 12px;border-radius:8px;min-width:220px}
        select:focus{outline:none;box-shadow:0 0 0 3px rgba(0,255,133,0.08);border-color:${COLORS.green}}
        .vs{color:${COLORS.green};font-weight:800}
        .predict-btn{background:${COLORS.green};color:#071019;padding:12px 20px;border-radius:10px;border:none;font-weight:800;cursor:pointer}
        .predict-btn:disabled{opacity:0.6;cursor:not-allowed}
        .card{background:${COLORS.card};padding:20px;border-radius:12px;margin-top:18px;text-align:left}
        .teams{display:flex;justify-content:space-between;align-items:center;font-size:20px;font-weight:800}
        .badge{display:inline-block;padding:6px 10px;border-radius:999px;background:${COLORS.green};color:#041018;font-weight:800}
        .prob-row{display:flex;align-items:center;gap:12px;margin-top:12px}
        .prob-label{width:120px}
        .bar-outer{flex:1;background:#071018;border-radius:8px;height:18px;overflow:hidden}
        .bar-inner{height:100%;transition:width 700ms ease}
        .prob-num{width:56px;text-align:right}
        .empty{color:#9aa6b2;padding:28px}
        .loading{color:#9aa6b2}
        .error{color:${COLORS.red};margin-top:12px}
        .sim-header{margin-bottom:32px}
        .sim-heading{font-size:24px;font-weight:900;margin-bottom:8px}
        .sim-subtitle{font-size:12px;color:#b8c2cc;margin-bottom:24px}
        .sim-table{width:100%;border-collapse:collapse;font-size:13px}
        .sim-table td{padding:12px;text-align:right;border-bottom:1px solid #1f2937}
        .sim-table td:first-child{text-align:left}
        .sim-table tr{background:${COLORS.card}}
        .bar-inline{display:inline-block;height:14px;border-radius:4px;background:${COLORS.green};margin:0 8px}
      `}</style>

      <div className="container">
        <div className="header">
          <span style={{fontSize:32}}>⚽</span>
          <div>
            <div style={{display:'flex',alignItems:'center',gap:8}}>
              <div style={{fontSize:20}}>Premier League</div>
              <div style={{fontSize:22}} className="accent">Hub</div>
            </div>
            <div className="subtitle">2024/25 · Data & Predictions</div>
          </div>
        </div>

        <div className="tab-switcher">
          <button 
            className={`tab-btn ${activeTab === 'predictor' ? 'active' : ''}`}
            onClick={() => setActiveTab('predictor')}
          >
            Match Predictor
          </button>
          <button 
            className={`tab-btn ${activeTab === 'simulator' ? 'active' : ''}`}
            onClick={() => setActiveTab('simulator')}
          >
            Season Simulator
          </button>
        </div>

        <div style={{marginTop:24}}>
          {activeTab === 'predictor' ? (
            <>
              {loadingTeams ? <div className="loading">Loading teams...</div> : null}
              {teamsError ? <div className="error">{teamsError}</div> : null}

              <div className="selector">
                <select value={home} onChange={e => setHome(e.target.value)}>
                  <option value="">Home Team</option>
                  {teams.map(t => <option key={t} value={t}>{t}</option>)}
                </select>

                <div className="vs">VS</div>

                <select value={away} onChange={e => setAway(e.target.value)}>
                  <option value="">Away Team</option>
                  {teams.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>

              <div>
                <button className="predict-btn" onClick={predict} disabled={predictLoading || loadingTeams}>
                  {predictLoading ? 'Predicting...' : 'Predict Match'}
                </button>
              </div>

              {error ? <div className="error">{error}</div> : null}

              <div style={{marginTop:18}}>
                {!result ? emptyState : (
                  <div className="card">
                    <div className="teams">
                      <div>{result.home_team}</div>
                      <div style={{textAlign:'center'}}>
                        <div style={{fontSize:12,color:'#94a3b8'}}>Predicted</div>
                        <div style={{marginTop:6}} className="badge">{result.predicted_result_label}</div>
                      </div>
                      <div style={{textAlign:'right'}}>{result.away_team}</div>
                    </div>

                    <div style={{marginTop:12}}>
                      {bar('Home Win', (result.home_win_prob || 0) * 100, COLORS.green)}
                      {bar('Draw', (result.draw_prob || 0) * 100, COLORS.yellow)}
                      {bar('Away Win', (result.away_win_prob || 0) * 100, COLORS.red)}
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="sim-header">
                <div className="sim-heading">2025/26 Pre-Season Prediction</div>
                <div className="sim-subtitle">Based on end of 2024/25 Elo ratings · 10,000 simulations</div>
              </div>

              <div style={{marginBottom:24}}>
                <button className="predict-btn" onClick={runSimulation} disabled={simulationLoading}>
                  {simulationLoading ? 'Running...' : 'Run Simulation'}
                </button>
              </div>

              {simulationLoading && <div className="loading">Running 10,000 simulations...</div>}

              {simulationData && (
                <table className="sim-table">
                  <thead>
                    <tr style={{fontWeight:800,background:'#0a0e12'}}>
                      <td>Pos</td>
                      <td>Team</td>
                      <td>Title%</td>
                      <td>Top 4%</td>
                      <td>Top 6%</td>
                      <td>Rel%</td>
                      <td style={{textAlign:'right'}}>Avg Pos</td>
                    </tr>
                  </thead>
                  <tbody>
                    {simulationData.map((row) => (
                      <tr key={row.position} style={{background: getRowColor(row.position)}}>
                        <td style={{textAlign:'left',fontWeight:800}}>{row.position}</td>
                        <td style={{textAlign:'left'}}>{row.team}</td>
                        <td>
                          <div style={{display:'flex',alignItems:'center',justifyContent:'flex-end'}}>
                            <span style={{marginRight:6}}>{row.title_prob}%</span>
                            <div className="bar-inline" style={{width: row.title_prob * 1.5 + 'px'}}></div>
                          </div>
                        </td>
                        <td>{row.top4_prob}%</td>
                        <td>{row.top6_prob}%</td>
                        <td>{row.relegation_prob}%</td>
                        <td>{row.avg_position}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
