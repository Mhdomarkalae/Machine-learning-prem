import React, { useEffect, useRef, useState } from 'react'

const PLBadge = () => (
  <div
    style={{
      width: '100%',
      maxWidth: '320px',
      maxHeight: '320px',
      aspectRatio: '0.86 / 1',
      position: 'relative',
      display: 'grid',
      placeItems: 'center',
      filter: 'drop-shadow(0 0 40px rgba(0,255,135,0.3))',
    }}
  >
    <div
      style={{
        position: 'absolute',
        inset: 0,
        clipPath: 'polygon(50% 0%, 88% 10%, 100% 28%, 100% 66%, 50% 100%, 0% 66%, 0% 28%, 12% 10%)',
        background: 'linear-gradient(180deg, #0f1f3d 0%, #0a1628 100%)',
        border: '2px solid #00ff87',
        boxShadow: '0 0 40px rgba(0,255,135,0.3)',
      }}
    />
    <div
      style={{
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        transform: 'translateY(-8px)',
      }}
    >
      <div style={{ fontSize: 'clamp(86px, 10vw, 136px)', fontWeight: 900, lineHeight: 0.92, letterSpacing: '-0.08em' }}>
        <span style={{ color: '#ffffff' }}>P</span>
        <span style={{ color: '#00ff87' }}>L</span>
      </div>
      <div style={{ color: '#00ff87', fontSize: 'clamp(12px, 1.6vw, 15px)', fontWeight: 800, letterSpacing: '0.42em', marginTop: '10px' }}>
        PREDICT
      </div>
    </div>
  </div>
)

const COLORS = {
  bg: '#0a1628',
  hero: '#00a389',
  accent: '#00ff87',
  white: '#ffffff',
  card: '#0f1f3d',
  border: '#1e3a5f',
  tealTint: 'rgba(0, 163, 137, 0.16)',
  blueTint: 'rgba(32, 78, 172, 0.16)',
  redTint: 'rgba(164, 42, 42, 0.18)',
  yellow: '#ffd166',
  red: '#ff4d4f',
  muted: '#b7c6de',
}

const TEAM_BADGE_PAGES = {
  Arsenal: 'Arsenal_F.C.',
  'Aston Villa': 'Aston_Villa_F.C.',
  Bournemouth: 'A.F.C._Bournemouth',
  Brentford: 'Brentford_F.C.',
  Brighton: 'Brighton_%26_Hove_Albion_F.C.',
  Burnley: 'Burnley_F.C.',
  Chelsea: 'Chelsea_F.C.',
  'Crystal Palace': 'Crystal_Palace_F.C.',
  Everton: 'Everton_F.C.',
  Fulham: 'Fulham_F.C.',
  Leeds: 'Leeds_United_F.C.',
  Leicester: 'Leicester_City_F.C.',
  Liverpool: 'Liverpool_F.C.',
  'Man City': 'Manchester_City_F.C.',
  'Man United': 'Manchester_United_F.C.',
  Newcastle: 'Newcastle_United_F.C.',
  'Nott\'m Forest': 'Nottingham_Forest_F.C.',
  Tottenham: 'Tottenham_Hotspur_F.C.',
  'West Ham': 'West_Ham_United_F.C.',
  Wolves: 'Wolverhampton_Wanderers_F.C.',
  Sunderland: 'Sunderland_A.F.C.',
}

const TEAM_BADGE_FALLBACK_COLORS = {
  Arsenal: '#ef0107',
  'Aston Villa': '#670e36',
  Bournemouth: '#d3172c',
  Brentford: '#e30613',
  Brighton: '#0057b8',
  Burnley: '#6c1d45',
  Chelsea: '#034694',
  'Crystal Palace': '#1b458f',
  Everton: '#003399',
  Fulham: '#ffffff',
  Leeds: '#ffcd00',
  Leicester: '#003090',
  Liverpool: '#c8102e',
  'Man City': '#6cabdd',
  'Man United': '#da291c',
  Newcastle: '#241f20',
  'Nott\'m Forest': '#dd0000',
  Tottenham: '#132257',
  'West Ham': '#7a263a',
  Wolves: '#fdb913',
  Sunderland: '#eb172b',
}

const TEAM_BADGE_SIZE = 22

const TEAM_BADGES = {
  Arsenal: '🔴',
  Chelsea: '🔵',
  Liverpool: '🔴',
  'Man City': '🔵',
  'Man United': '🔴',
  Tottenham: '⚪',
  'Aston Villa': '🟣',
  Newcastle: '⚫',
  Brighton: '🔵',
  'West Ham': '🟣',
  Wolves: '🟡',
  Everton: '🔵',
  Brentford: '🔴',
  Fulham: '⚪',
  'Crystal Palace': '🔴',
  Bournemouth: '🔴',
  'Nott\'m Forest': '🔴',
  Leeds: '⚪',
  Burnley: '🟣',
  Sunderland: '🔴',
  Coventry: '🔵',
  Ipswich: '🔵',
  Southampton: '🔴',
  Leicester: '🔵',
}

const TEAM_LOGOS = {
  'Arsenal': 'https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg',
  'Chelsea': 'https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg',
  'Liverpool': 'https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg',
  'Man City': 'https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg',
  'Man United': 'https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg',
  'Tottenham': 'https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg',
  'Aston Villa': 'https://upload.wikimedia.org/wikipedia/en/f/f9/Aston_Villa_FC_crest_%282016%29.svg',
  'Newcastle': 'https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg',
  'Brighton': 'https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg',
  'West Ham': 'https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg',
  'Everton': 'https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg',
  'Brentford': 'https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg',
  'Fulham': 'https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg',
  'Crystal Palace': 'https://upload.wikimedia.org/wikipedia/en/0/0c/Crystal_Palace_FC_logo.svg',
  'Bournemouth': 'https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg',
  "Nott'm Forest": 'https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg',
  'Leeds': 'https://upload.wikimedia.org/wikipedia/en/5/54/Leeds_United_F.C._logo.svg',
  'Burnley': 'https://upload.wikimedia.org/wikipedia/en/6/62/Burnley_F.C._Logo.svg',
  'Sunderland': 'https://upload.wikimedia.org/wikipedia/en/7/77/Logo_Sunderland.svg',
  'Coventry': 'https://upload.wikimedia.org/wikipedia/en/8/8e/Coventry_City_FC.svg',
  'Ipswich': 'https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg',
  'Southampton': 'https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg',
}

const TEAM_LOGO_FALLBACK = 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 40 40%22%3E%3Ccircle cx=%2220%22 cy=%2220%22 r=%2218%22 fill=%22%23162040%22 stroke=%22%2300a389%22 stroke-width=%222%22/%3E%3Ctext x=%2250%25%22 y=%2255%25%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22 fill=%22white%22 font-size=%2214%22 font-family=%22Inter%22%3E%E2%9A%BD%3C/text%3E%3C/svg%3E'

const getTeamInitials = (team) => {
  if (team === "Nott'm Forest") return 'NFO'
  if (team === 'Man City') return 'MC'
  if (team === 'Man United') return 'MU'
  return team
    .split(/\s+/)
    .map((part) => part[0])
    .join('')
    .slice(0, 3)
    .toUpperCase()
}

const getTeamBadgeFallback = (team) => {
  const background = TEAM_BADGE_FALLBACK_COLORS[team] || COLORS.card
  const textColor = team === 'Fulham' ? COLORS.bg : COLORS.white
  return {
    background,
    color: textColor,
    initials: getTeamInitials(team),
  }
}

export default function App() {
  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
  const predictorRef = useRef(null)
  const table25Ref = useRef(null)
  const table26Ref = useRef(null)

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
  const [simulationData2627, setSimulationData2627] = useState(null)
  const [simulationLoading2627, setSimulationLoading2627] = useState(false)
  const [teamBadgeUrls, setTeamBadgeUrls] = useState({})

  useEffect(() => {
    let mounted = true

    async function fetchTeams() {
      try {
        setLoadingTeams(true)
        const res = await fetch(`${API_BASE}/teams`)
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

  useEffect(() => {
    let cancelled = false

    async function loadTeamBadges() {
      const badgeEntries = await Promise.all(
        Object.entries(TEAM_BADGE_PAGES).map(async ([team, pageTitle]) => {
          try {
            const res = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(pageTitle)}`)
            if (!res.ok) return [team, null]
            const data = await res.json()
            return [team, data.originalimage?.source || data.thumbnail?.source || null]
          } catch {
            return [team, null]
          }
        }),
      )

      if (!cancelled) {
        setTeamBadgeUrls(Object.fromEntries(badgeEntries.filter(([, url]) => Boolean(url))))
      }
    }

    loadTeamBadges()

    return () => {
      cancelled = true
    }
  }, [])

  function scrollToSection(ref) {
    if (ref?.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

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
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home_team: home, away_team: away }),
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
      setSimulationData(null)
      setSimulationLoading(true)
      const res = await fetch(`${API_BASE}/simulate`)
      if (!res.ok) throw new Error('Simulation failed')
      const data = await res.json()
      setSimulationData(data.table)
    } catch (err) {
      setError(err.message)
    } finally {
      setSimulationLoading(false)
    }
  }

  async function runSimulation2627() {
    try {
      setSimulationData2627(null)
      setSimulationLoading2627(true)
      const res = await fetch(`${API_BASE}/simulate2627`)
      if (!res.ok) throw new Error('Simulation failed')
      const data = await res.json()
      setSimulationData2627(data.table)
    } catch (err) {
      setError(err.message)
    } finally {
      setSimulationLoading2627(false)
    }
  }

  const getRowClass = (position) => {
    return 'row-default'
  }

  const getRelegationStart = (totalTeams) => (totalTeams === 19 ? 17 : 18)

  const getRowAccentColor = (position, totalTeams) => {
    const relegationStart = getRelegationStart(totalTeams)

    if (position === 1) return '#d4af37'
    if (position >= 2 && position <= 4) return '#4ea1ff'
    if (position >= 5 && position <= 6) return '#ff9f1c'
    if (position >= relegationStart) return COLORS.red
    return 'rgba(255, 255, 255, 0.16)'
  }

  const getRowMarker = (position, totalTeams) => {
    const relegationStart = getRelegationStart(totalTeams)

    if (position === 1) return '🏆'
    if (position >= 2 && position <= 4) return '🔵'
    if (position >= 5 && position <= 6) return '🟠'
    if (position >= relegationStart) return '🔴'
    return ''
  }

  const getEloColor = (elo) => {
    if (elo > 1550) return COLORS.accent
    if (elo >= 1450) return COLORS.yellow
    return COLORS.red
  }

  const getDisplayedTitleProb = (row) => row.title_prob ?? 0
  const getDisplayedTop4Prob = (row) => row.top4_prob ?? 0
  const getDisplayedTop6Prob = (row) => row.top6_prob ?? 0
  const getDisplayedRelProb = (row) => row.relegation_prob ?? 0

  const renderTeamBadge = (team) => {
    const badgeUrl = teamBadgeUrls[team]

    if (badgeUrl) {
      return <img className="team-badge" src={badgeUrl} alt="" aria-hidden="true" />
    }

    const fallback = getTeamBadgeFallback(team)
    return <span className="team-badge team-badge-fallback" style={{ background: fallback.background, color: fallback.color }}>{fallback.initials}</span>
  }

  const renderTable = (rows) => (
    <div>
      <div className="table-shell">
        <table style={{width:'100%', borderCollapse:'collapse'}}>
          <thead>
            <tr style={{backgroundColor:'#0a1628', borderBottom:'2px solid #1e3a5f'}}>
              <th style={{padding:'12px 16px', textAlign:'left', color:'#8899aa', fontSize:'11px', fontWeight:'700', letterSpacing:'1px', width:'60px'}}>POS</th>
              <th style={{padding:'12px 16px', textAlign:'left', color:'#8899aa', fontSize:'11px', fontWeight:'700', letterSpacing:'1px'}}>TEAM</th>
              <th style={{padding:'12px 16px', textAlign:'center', color:'#8899aa', fontSize:'11px', fontWeight:'700', letterSpacing:'1px'}}>ELO</th>
              <th style={{padding:'12px 16px', textAlign:'center', color:'#8899aa', fontSize:'11px', fontWeight:'700', letterSpacing:'1px'}}>TITLE %</th>
              <th style={{padding:'12px 16px', textAlign:'center', color:'#8899aa', fontSize:'11px', fontWeight:'700', letterSpacing:'1px'}}>TOP 4 %</th>
              <th style={{padding:'12px 16px', textAlign:'center', color:'#8899aa', fontSize:'11px', fontWeight:'700', letterSpacing:'1px'}}>TOP 6 %</th>
              <th style={{padding:'12px 16px', textAlign:'center', color:'#8899aa', fontSize:'11px', fontWeight:'700', letterSpacing:'1px'}}>REL %</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const pos = row.position
              const isTitle = pos === 1
              const isCL = pos >= 2 && pos <= 4
              const isEL = pos >= 5 && pos <= 6
              const totalTeams = rows.length
              const isRel = pos >= totalTeams - 2
              const borderColor = isTitle ? '#ffd700' : isCL ? '#4a9eff' : isEL ? '#ff9500' : isRel ? '#ff4444' : 'transparent'
              const posIcon = isTitle ? '🏆' : isCL ? '🔵' : isEL ? '🟠' : isRel ? '🔴' : ''
              const eloBarColor = row.elo > 1550 ? '#00ff87' : row.elo > 1450 ? '#ffd700' : '#ff4444'
              const eloBarWidth = Math.max(10, Math.min(100, ((row.elo - 1200) / 500) * 100))
              return (
                <tr key={row.team} style={{
                  backgroundColor: index % 2 === 0 ? '#0f1f3d' : '#0a1628',
                  borderLeft: `3px solid ${borderColor}`,
                  borderBottom: '1px solid #1e3a5f',
                }}>
                  <td style={{padding:'14px 16px', color:'#ffffff', fontWeight:'700', fontSize:'14px'}}>
                    {posIcon} {pos}
                  </td>
                  <td style={{padding:'14px 16px'}}>
                    <div style={{display:'flex', alignItems:'center', gap:'10px'}}>
                      <img
                        src={TEAM_LOGOS[row.team] || TEAM_LOGO_FALLBACK}
                        alt={row.team}
                        style={{width:'28px', height:'28px', objectFit:'contain'}}
                        onError={(e) => {
                          e.currentTarget.onerror = null
                          e.currentTarget.src = TEAM_LOGO_FALLBACK
                        }}
                      />
                      <div>
                        <div style={{color:'#ffffff', fontWeight:'700', fontSize:'15px'}}>{row.team}</div>
                        <div style={{display:'flex', alignItems:'center', gap:'8px', marginTop:'4px'}}>
                          <span style={{color:'#8899aa', fontSize:'12px'}}>{row.elo}</span>
                          <div style={{height:'4px', width:`${Math.max(10, Math.min(100, ((row.elo - 1200) / 500) * 100))}px`, backgroundColor: row.elo > 1550 ? '#00ff87' : row.elo > 1450 ? '#ffd700' : '#ff4444', borderRadius:'2px'}}></div>
                        </div>
                      </div>
                    </div>
                  </td>
                  <td style={{padding:'14px 16px', textAlign:'center', color:'#ffffff', fontSize:'14px'}}>{row.elo}</td>
                  <td style={{padding:'14px 16px', textAlign:'center', color:'#ffffff', fontSize:'14px', fontWeight:'600'}}>{row.title_prob}%</td>
                  <td style={{padding:'14px 16px', textAlign:'center', color:'#ffffff', fontSize:'14px'}}>{row.top5_prob}%</td>
                  <td style={{padding:'14px 16px', textAlign:'center', color:'#ffffff', fontSize:'14px'}}>{row.top6_prob}%</td>
                  <td style={{padding:'14px 16px', textAlign:'center', color: isRel ? '#ff4444' : '#ffffff', fontSize:'14px'}}>{row.relegation_prob}%</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="table-legend">🏆 1st place&nbsp;&nbsp;&nbsp;🔵 2-4&nbsp;&nbsp;&nbsp;🟠 5-6&nbsp;&nbsp;&nbsp;🔴 Relegation spots</div>
    </div>
  )

  // Flag a team whose most recent match predates the dataset max by >12 months.
  // The year comes from the /predict response, not a hardcoded value.
  const staleYear = (matchDate, maxDate) => {
    if (!matchDate || !maxDate) return null
    const d = new Date(matchDate)
    const max = new Date(maxDate)
    if (Number.isNaN(d.getTime()) || Number.isNaN(max.getTime())) return null
    const months = (max - d) / (1000 * 60 * 60 * 24 * 30.44)
    return months > 12 ? d.getFullYear() : null
  }
  const homeStaleYear = result ? staleYear(result.home_last_match_date, result.data_max_date) : null
  const awayStaleYear = result ? staleYear(result.away_last_match_date, result.data_max_date) : null

  return (
    <div className="page">
      <style>{`
        :root { color-scheme: dark; }
        * { box-sizing: border-box; }
        html { scroll-behavior: smooth; }
        body { margin: 0; background: ${COLORS.bg}; }
        .page { min-height: 100vh; background: ${COLORS.bg}; color: ${COLORS.white}; font-family: Inter, system-ui, sans-serif; }
        .nav { position: sticky; top: 0; z-index: 30; width: 100%; background: rgba(10, 22, 40, 0.96); backdrop-filter: blur(14px); border-bottom: 1px solid rgba(30, 58, 95, 0.8); }
        .nav-inner { max-width: 1320px; margin: 0 auto; padding: 18px 24px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }
        .brand { display: flex; align-items: center; gap: 12px; font-weight: 900; font-size: 22px; letter-spacing: 0.2px; }
        .brand-mark { width: 38px; height: 38px; border-radius: 12px; display: grid; place-items: center; background: linear-gradient(135deg, ${COLORS.hero}, ${COLORS.accent}); color: #05111f; box-shadow: 0 10px 30px rgba(0, 255, 135, 0.18); }
        .nav-links { display: flex; align-items: center; gap: 26px; flex-wrap: wrap; justify-content: center; }
        .nav-link { border: 0; background: transparent; color: ${COLORS.white}; font-weight: 700; cursor: pointer; opacity: 0.9; padding: 6px 0; position: relative; }
        .nav-link::after { content: ''; position: absolute; left: 0; bottom: -6px; width: 100%; height: 2px; background: ${COLORS.accent}; transform: scaleX(0); transform-origin: left; transition: transform 180ms ease; }
        .nav-link:hover::after { transform: scaleX(1); }
        .search-pill { width: 42px; height: 42px; border-radius: 999px; display: grid; place-items: center; border: 1px solid rgba(255,255,255,0.12); color: ${COLORS.white}; background: rgba(255,255,255,0.03); }
        .hero { width: 100%; background: #0a1628; overflow: hidden; min-height: 100vh; display: flex; align-items: center; }
        .hero-inner { max-width: 1440px; width: 100%; margin: 0 auto; padding: 80px 60px; display: flex; align-items: center; justify-content: space-between; gap: 40px; min-height: 100vh; }
        .hero-copy { flex: 0 0 55%; max-width: 55%; text-align: left; }
        .hero-visual { flex: 0 0 45%; max-width: 45%; display: flex; justify-content: center; align-items: center; }
        .hero-pill { display: inline-flex; align-items: center; gap: 10px; padding: 10px 16px; border-radius: 999px; border: 1px solid #00ff87; color: #bfe9dd; background: rgba(255,255,255,0.02); font-size: 13px; font-weight: 800; letter-spacing: 0.3px; margin-bottom: 24px; }
        .hero-heading { margin: 0; line-height: 0.9; font-weight: 900; letter-spacing: -0.08em; text-transform: uppercase; }
        .hero-heading-line1 { display: block; color: #ffffff; font-size: clamp(72px, 8vw, 80px); }
        .hero-heading-line2 { display: block; font-size: clamp(72px, 8vw, 80px); margin-top: 6px; }
        .hero-subtitle { margin: 18px 0 28px; max-width: 640px; font-size: 19px; line-height: 1.55; color: rgba(255,255,255,0.88); font-weight: 500; }
        .powered-by { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 30px; }
        .powered-pill { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 20px; background: rgba(15, 31, 61, 0.9); border: 1px solid rgba(0,255,135,0.22); color: ${COLORS.white}; font-size: 12.5px; font-weight: 700; line-height: 1.2; white-space: nowrap; }
        .hero-btn { border: 1px solid #ffffff; background: transparent; color: #ffffff; font-weight: 900; letter-spacing: 1.8px; padding: 14px 24px; border-radius: 999px; cursor: pointer; transition: transform 180ms ease, background 180ms ease, color 180ms ease; }
        .hero-btn:hover { transform: translateY(-1px); background: #ffffff; color: #0a1628; }
        .content { max-width: 1320px; margin: 0 auto; padding: 48px 24px 88px; }
        .section { margin-top: 34px; padding: 28px; background: rgba(15, 31, 61, 0.58); border: 1px solid rgba(30, 58, 95, 0.9); border-radius: 28px; box-shadow: 0 24px 80px rgba(0, 0, 0, 0.18); }
        .section-header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 24px; }
        .section-title { margin: 0; font-size: 32px; line-height: 1; font-weight: 900; letter-spacing: -0.8px; }
        .section-title span { color: ${COLORS.hero}; }
        .section-underline { width: 92px; height: 4px; border-radius: 999px; background: ${COLORS.accent}; margin-top: 12px; box-shadow: 0 0 24px rgba(0,255,135,0.45); }
        .section-subtitle { color: ${COLORS.muted}; font-size: 14px; margin: 0; }
        .error { margin-top: 18px; color: ${COLORS.red}; font-weight: 700; }
        .selector-grid { display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; align-items: stretch; margin-bottom: 20px; }
        .team-card { background: linear-gradient(180deg, rgba(10, 22, 40, 0.92), rgba(15, 31, 61, 0.92)); border: 1px solid ${COLORS.border}; border-radius: 22px; padding: 18px; }
        .team-card label { display: block; text-align: left; color: ${COLORS.muted}; font-size: 12px; font-weight: 800; letter-spacing: 1px; margin-bottom: 10px; text-transform: uppercase; }
        .team-card select { width: 100%; appearance: none; border: 1px solid rgba(30, 58, 95, 0.95); background: rgba(8, 16, 30, 0.92); color: ${COLORS.white}; border-radius: 16px; padding: 16px 16px; font-size: 15px; font-weight: 700; }
        .team-card select:focus { outline: none; border-color: ${COLORS.accent}; box-shadow: 0 0 0 3px rgba(0, 255, 135, 0.13); }
        .vs-wrap { display: grid; place-items: center; min-width: 72px; }
        .vs { width: 60px; height: 60px; border-radius: 999px; display: grid; place-items: center; font-size: 18px; font-weight: 900; color: ${COLORS.accent}; border: 1px solid rgba(0,255,135,0.45); background: rgba(0, 255, 135, 0.06); box-shadow: 0 0 0 6px rgba(0,255,135,0.04); }
        .cta-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
        .predict-btn, .run-btn { background: linear-gradient(135deg, ${COLORS.hero}, #008c78); color: ${COLORS.white}; padding: 14px 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); font-weight: 900; letter-spacing: 0.6px; cursor: pointer; transition: transform 180ms ease, box-shadow 180ms ease, opacity 180ms ease; }
        .predict-btn:hover, .run-btn:hover { transform: translateY(-1px); box-shadow: 0 18px 30px rgba(0, 163, 137, 0.18); }
        .predict-btn:disabled, .run-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .mini-muted { color: ${COLORS.muted}; font-size: 13px; }
        .loading { color: ${COLORS.muted}; margin-top: 14px; }
        .result-card { margin-top: 24px; background: linear-gradient(180deg, rgba(10,22,40,0.88), rgba(15,31,61,0.96)); border: 1px solid rgba(30,58,95,0.95); border-radius: 24px; padding: 22px; }
        .result-top { display: grid; grid-template-columns: 1fr auto 1fr; gap: 18px; align-items: center; }
        .result-team { font-size: clamp(22px, 2.8vw, 34px); font-weight: 900; letter-spacing: -0.6px; }
        .result-center { text-align: center; }
        .result-label { display: inline-flex; align-items: center; justify-content: center; min-width: 110px; padding: 9px 14px; border-radius: 999px; background: ${COLORS.accent}; color: #021014; font-weight: 900; letter-spacing: 1px; }
        .prob-list { margin-top: 22px; display: grid; gap: 14px; }
        .prob-row { display: grid; grid-template-columns: 140px 1fr 58px; gap: 12px; align-items: center; }
        .prob-label { color: ${COLORS.white}; font-weight: 800; }
        .bar-outer { width: 100%; height: 16px; border-radius: 999px; overflow: hidden; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.06); }
        .bar-inner { height: 100%; border-radius: 999px; transition: width 650ms ease; }
        .prob-num { text-align: right; font-weight: 800; }
        .table-shell { overflow-x: auto; border-radius: 22px; border: 1px solid rgba(30,58,95,0.95); box-shadow: 0 18px 50px rgba(0,0,0,0.24); }
        .season-table { width: 100%; border-collapse: separate; border-spacing: 0; min-width: 900px; table-layout: fixed; background: rgba(10,22,40,0.72); }
        .season-table thead th { position: sticky; top: 0; z-index: 2; background: rgba(8, 16, 30, 0.98); color: ${COLORS.white}; text-align: left; padding: 16px 14px; font-size: 13px; font-weight: 900; letter-spacing: 0.8px; text-transform: uppercase; border-bottom: 1px solid rgba(30,58,95,0.95); }
        .season-table tbody td { padding: 12px 14px; border-bottom: 1px solid rgba(30,58,95,0.5); color: #c3cfdf; font-weight: 600; vertical-align: middle; white-space: nowrap; }
        .season-table th:nth-child(1), .season-table td:nth-child(1) { width: 72px; }
        .season-table th:nth-child(2), .season-table td:nth-child(2) { width: 260px; }
        .season-table th:nth-child(3), .season-table td:nth-child(3) { width: 150px; }
        .season-table th:nth-child(4), .season-table td:nth-child(4),
        .season-table th:nth-child(5), .season-table td:nth-child(5),
        .season-table th:nth-child(6), .season-table td:nth-child(6),
        .season-table th:nth-child(7), .season-table td:nth-child(7) { width: 106px; }
        .season-table tbody tr:nth-child(odd) { background: rgba(255,255,255,0.03); }
        .season-table tbody tr:nth-child(even) { background: rgba(255,255,255,0.015); }
        .season-table tbody tr:hover { background: rgba(255,255,255,0.045); }
        .season-table tbody tr td:first-child { border-left: 4px solid var(--row-accent, rgba(255,255,255,0.16)); }
        .pos-cell { color: #e6edf7; font-weight: 800; white-space: nowrap; }
        .team-cell { display: flex; align-items: center; gap: 10px; color: ${COLORS.white}; font-weight: 900; white-space: nowrap; min-width: 0; overflow: hidden; }
        .team-badge { width: ${TEAM_BADGE_SIZE}px; height: ${TEAM_BADGE_SIZE}px; border-radius: 50%; flex: none; object-fit: contain; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10); box-shadow: 0 0 0 1px rgba(0,0,0,0.14) inset; }
        .team-badge-fallback { display: grid; place-items: center; font-size: 9px; font-weight: 900; letter-spacing: 0.2px; }
        .team-name { overflow: hidden; text-overflow: ellipsis; }
        .elo-cell { display: flex; flex-direction: column; align-items: flex-start; gap: 6px; color: #d6e0ee; white-space: nowrap; width: 100%; overflow: hidden; justify-content: flex-start; }
        .elo-num { min-width: 44px; }
        .elo-bar-track { width: 72px; height: 7px; border-radius: 999px; overflow: hidden; background: rgba(255,255,255,0.08); flex: none; }
        .elo-bar-fill { display: block; height: 100%; border-radius: 999px; }
        .table-legend { margin-top: 12px; color: ${COLORS.muted}; font-size: 13px; font-weight: 700; }
        .footer-spacer { height: 18px; }
        .site-footer { margin-top: 18px; padding: 18px 16px 22px; border-top: 1px solid ${COLORS.hero}; color: #92a3be; text-align: center; font-size: 12.5px; line-height: 1.6; }
        .info-box { margin: 0 0 32px; padding: 18px 20px; background: rgba(0, 163, 137, 0.12); border-left: 3px solid ${COLORS.hero}; border-radius: 12px; color: ${COLORS.muted}; font-size: 13.5px; line-height: 1.6; }
        .info-box strong { color: ${COLORS.white}; }
        .model-panel { margin-top: 0; }
        .metric-tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
        .metric-tile { background: linear-gradient(180deg, rgba(10, 22, 40, 0.92), rgba(15, 31, 61, 0.92)); border: 1px solid ${COLORS.border}; border-radius: 22px; padding: 22px 20px; text-align: left; }
        .metric-label { color: ${COLORS.muted}; font-size: 12px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 10px; }
        .metric-value { color: ${COLORS.white}; font-size: clamp(40px, 5.5vw, 56px); font-weight: 900; line-height: 1; letter-spacing: -1.4px; }
        .metric-sub { color: ${COLORS.muted}; font-size: 13px; font-weight: 600; margin-top: 10px; line-height: 1.4; }
        .model-pills { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
        .model-pill { display: inline-flex; align-items: center; padding: 6px 14px; border-radius: 20px; background: rgba(15, 31, 61, 0.9); border: 1px solid rgba(0, 255, 135, 0.22); color: ${COLORS.white}; font-size: 12.5px; font-weight: 700; line-height: 1.2; white-space: nowrap; }
        .model-lead { margin: 2px 0 22px; max-width: 940px; font-size: clamp(17px, 2.1vw, 22px); font-weight: 700; line-height: 1.5; letter-spacing: -0.2px; color: ${COLORS.muted}; }
        .model-lead .lead-hl { color: ${COLORS.accent}; font-weight: 900; }
        .model-lead .lead-em { color: ${COLORS.white}; font-weight: 800; }
        .data-note { margin-top: 16px; color: ${COLORS.muted}; font-size: 12.5px; line-height: 1.6; }
        .stale-warn { margin-top: 14px; padding: 11px 14px; background: ${COLORS.redTint}; border-left: 3px solid ${COLORS.red}; border-radius: 12px; color: #ffdada; font-size: 13px; font-weight: 600; line-height: 1.55; }
        .stale-warn strong { color: ${COLORS.white}; }
        .stale-warn > div + div { margin-top: 6px; }
        @media (max-width: 760px) { .metric-tiles { grid-template-columns: 1fr; } }
        @media (max-width: 1080px) {
          .hero-inner { flex-direction: column; align-items: flex-start; min-height: auto; }
          .hero-copy, .hero-visual { flex: 1 1 auto; max-width: 100%; width: 100%; }
          .hero-visual { justify-content: flex-start; }
          .selector-grid { grid-template-columns: 1fr; }
          .vs-wrap { min-height: 56px; }
          .result-top { grid-template-columns: 1fr; text-align: center; }
          .prob-row { grid-template-columns: 110px 1fr 48px; }
        }
        @media (max-width: 760px) {
          .nav-inner { padding: 14px 16px; }
          .nav-links { gap: 14px; font-size: 14px; }
          .hero-inner { padding: 28px 16px 40px; }
          .hero-heading-line1, .hero-heading-line2 { font-size: clamp(54px, 16vw, 72px); }
          .hero-subtitle { font-size: 16px; }
          .content { padding: 22px 16px 64px; }
          .section { padding: 20px 16px; border-radius: 22px; }
          .section-header { flex-direction: column; align-items: flex-start; }
          .section-title { font-size: 26px; }
        }
      `}</style>

      <header className="nav">
        <div className="nav-inner">
          <div className="brand">
            <div className="brand-mark">⚽</div>
            <div>PredictPL</div>
          </div>

          <nav className="nav-links" aria-label="Primary navigation">
            <button className="nav-link" onClick={() => scrollToSection(predictorRef)}>Home</button>
            <button className="nav-link" onClick={() => scrollToSection(predictorRef)}>Predict</button>
            <button className="nav-link" onClick={() => scrollToSection(table25Ref)}>Table 25/26</button>
            <button className="nav-link" onClick={() => scrollToSection(table26Ref)}>Table 26/27</button>
          </nav>

        </div>
      </header>

      <section className="hero">
        <div className="hero-inner">
          <div className="hero-copy">
            <h1 className="hero-heading">
              <span className="hero-heading-line1">PREDICT</span>
              <span className="hero-heading-line2">
                {"PREMIER LEAGUE".split("").map((char, i) => (
                  <span key={i} style={{ color: char === ' ' ? 'transparent' : i % 2 === 0 ? '#ffffff' : '#00ff87' }}>
                    {char === ' ' ? '\u00A0' : char}
                  </span>
                ))}
              </span>
            </h1>
            <p className="hero-subtitle">Match predictions · Season simulations · 10,000 Monte Carlo runs</p>
            <div className="powered-by" aria-label="Powered by technologies">
              <span className="powered-pill">⚡ XGBoost ML</span>
              <span className="powered-pill">📊 Elo Ratings</span>
              <span className="powered-pill">⚽ xG Data</span>
              <span className="powered-pill">📈 Form &amp; Momentum</span>
              <span className="powered-pill">🏠 Home &amp; Away Stats</span>
              <span className="powered-pill">🤝 Head-to-Head History</span>
              <span className="powered-pill">😴 Rest Days</span>
              <span className="powered-pill">🔥 Win Streaks</span>
              <span className="powered-pill">🛡️ Clean Sheet Rate</span>
              <span className="powered-pill">⚖️ Draw Tendency</span>
              <span className="powered-pill">🎯 Scoring Rate</span>
              <span className="powered-pill">🔄 Monte Carlo Simulation</span>
              <span className="powered-pill">📉 Bayesian Smoothing</span>
              <span className="powered-pill">⏱️ Time Decay</span>
            </div>
            <button className="hero-btn" onClick={() => scrollToSection(predictorRef)}>GET STARTED</button>
          </div>

          <div className="hero-visual">
            <PLBadge />
          </div>
        </div>
      </section>

      <main className="content">
        <section className="section model-panel">
          <div className="section-header">
            <div>
              <h2 className="section-title">About the <span>model</span></h2>
              <div className="section-underline" />
            </div>
          </div>

          <p className="model-lead">
            Trained on <span className="lead-hl">8,650</span> Premier League matches from <span className="lead-hl">2000/01 through 2022/23</span>. Tested on <span className="lead-hl">2023/24 and 2024/25</span> — <span className="lead-em">seasons the model never saw during training.</span>
          </p>

          <div className="metric-tiles">
            <div className="metric-tile">
              <div className="metric-label">Accuracy</div>
              <div className="metric-value">53.7%</div>
              <div className="metric-sub">vs 43.4% home-team baseline</div>
            </div>
            <div className="metric-tile">
              <div className="metric-label">Log loss</div>
              <div className="metric-value">0.989</div>
              <div className="metric-sub">lower is better</div>
            </div>
            <div className="metric-tile">
              <div className="metric-label">Test set</div>
              <div className="metric-value">760</div>
              <div className="metric-sub">held-out matches</div>
            </div>
          </div>

          <div className="model-pills">
            <span className="model-pill">XGBoost</span>
            <span className="model-pill">40 features</span>
            <span className="model-pill">Leak-audited</span>
          </div>
        </section>

        <div className="info-box">
          <strong>How it works:</strong> The 2025/26 table uses data from before this season started. The 2026/27 table is a separate preview built on 2025/26 data. Each prediction is made with only information available before those matches kick off—no data leakage.
        </div>

        <section ref={predictorRef} id="predictor" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">Match <span>Predictor</span></h2>
              <div className="section-underline" />
            </div>
            <p className="section-subtitle">Select two teams and let the model estimate the outcome.</p>
          </div>

          {loadingTeams ? <div className="loading">Loading teams...</div> : null}
          {teamsError ? <div className="error">{teamsError}</div> : null}

          <div className="selector-grid">
            <div className="team-card">
              <label>Home Team</label>
              <select value={home} onChange={(e) => setHome(e.target.value)}>
                <option value="">Select home team</option>
                {teams.map((team) => (
                  <option key={team} value={team}>{team}</option>
                ))}
              </select>
            </div>

            <div className="vs-wrap"><div className="vs">VS</div></div>

            <div className="team-card">
              <label>Away Team</label>
              <select value={away} onChange={(e) => setAway(e.target.value)}>
                <option value="">Select away team</option>
                {teams.map((team) => (
                  <option key={team} value={team}>{team}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="cta-row">
            <button className="predict-btn" onClick={predict} disabled={predictLoading || loadingTeams}>
              {predictLoading ? 'Predicting...' : 'Predict Match'}
            </button>
            <div className="mini-muted">Powered by Elo, form, and xG features</div>
          </div>

          {error ? <div className="error">{error}</div> : null}

          {!result ? (
            <div className="result-card">
              <div className="mini-muted">Select two teams to get a prediction.</div>
            </div>
          ) : (
            <div className="result-card">
              <div className="result-top">
                <div className="result-team" style={{ textAlign: 'left' }}>{result.home_team}</div>
                <div className="result-center">
                  <div className="mini-muted">Predicted</div>
                  <div style={{ marginTop: 8 }} className="result-label">{result.predicted_result_label}</div>
                </div>
                <div className="result-team" style={{ textAlign: 'right' }}>{result.away_team}</div>
              </div>

              <div className="prob-list">
                <div className="prob-row">
                  <div className="prob-label">Home Win</div>
                  <div className="bar-outer"><div className="bar-inner" style={{ width: `${(result.home_win_prob || 0) * 100}%`, background: COLORS.accent }} /></div>
                  <div className="prob-num">{Math.round((result.home_win_prob || 0) * 100)}%</div>
                </div>
                <div className="prob-row">
                  <div className="prob-label">Draw</div>
                  <div className="bar-outer"><div className="bar-inner" style={{ width: `${(result.draw_prob || 0) * 100}%`, background: COLORS.yellow }} /></div>
                  <div className="prob-num">{Math.round((result.draw_prob || 0) * 100)}%</div>
                </div>
                <div className="prob-row">
                  <div className="prob-label">Away Win</div>
                  <div className="bar-outer"><div className="bar-inner" style={{ width: `${(result.away_win_prob || 0) * 100}%`, background: COLORS.red }} /></div>
                  <div className="prob-num">{Math.round((result.away_win_prob || 0) * 100)}%</div>
                </div>
              </div>

              {(homeStaleYear || awayStaleYear) ? (
                <div className="stale-warn">
                  {homeStaleYear ? (
                    <div>⚠️ <strong>{result.home_team}</strong> last appears in the data in {homeStaleYear} — it is evaluated on that season&apos;s form, not current.</div>
                  ) : null}
                  {awayStaleYear ? (
                    <div>⚠️ <strong>{result.away_team}</strong> last appears in the data in {awayStaleYear} — it is evaluated on that season&apos;s form, not current.</div>
                  ) : null}
                </div>
              ) : null}

              <div className="data-note">
                Features come from each team&apos;s most recent appearance in the dataset, not a date you pick. Currently active teams use recent form (data runs through May 2026); teams no longer in the Premier League fall back to their last top-flight season — so a club like Blackburn is evaluated on ~2012 form.
              </div>
            </div>
          )}
        </section>

        <div className="footer-spacer" />

        <section ref={table25Ref} id="table-25-26" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">2025/26 <span>Pre-Season Prediction</span></h2>
              <div className="section-underline" />
            </div>
            <p className="section-subtitle">Based on end of 2024/25 Elo ratings · 10,000 simulations</p>
          </div>

          <div className="cta-row">
            <button className="run-btn" onClick={runSimulation} disabled={simulationLoading}>
              {simulationLoading ? 'Running...' : 'Run Simulation'}
            </button>
            <div className="mini-muted">Rows are highlighted by qualification zone automatically.</div>
          </div>

          {simulationLoading && <div className="loading">Running 10,000 simulations...</div>}
          {simulationData ? renderTable(simulationData) : null}
        </section>

        <div className="footer-spacer" />

        <section ref={table26Ref} id="table-26-27" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">2026/27 <span>Pre-Season Prediction</span></h2>
              <div className="section-underline" />
            </div>
            <p className="section-subtitle">Based on end of 2025/26 Elo ratings · 10,000 simulations</p>
          </div>

          <div className="mini-muted" style={{ marginBottom: 16 }}>
            The 2025/26 table shows this season&apos;s forecast. The 2026/27 table is a separate preview for next season.
          </div>

          <div className="cta-row">
            <button className="run-btn" onClick={runSimulation2627} disabled={simulationLoading2627}>
              {simulationLoading2627 ? 'Running...' : 'Run Simulation'}
            </button>
            <div className="mini-muted">Identical simulation layout, updated for the new season.</div>
          </div>

          {simulationLoading2627 && <div className="loading">Running 10,000 simulations...</div>}
          {simulationData2627 ? renderTable(simulationData2627) : null}
        </section>

        <footer className="site-footer">
          PredictPL · Built with XGBoost, FastAPI &amp; React · Data from football-data.co.uk &amp; Understat · UCF Computer Science
        </footer>
      </main>
    </div>
  )
}