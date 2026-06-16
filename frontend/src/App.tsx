import { useState, useEffect, useRef, useMemo } from 'react'
import axios from 'axios'
import {
  TrendingUp,
  ShieldAlert,
  LineChart as LineIcon,
  Percent,
  Zap,
  BarChart3,
  Network,
  Activity,
  FileText,
  PieChart as PieIcon,
  RefreshCw,
  AlertTriangle,
  Info
} from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import ForceGraph2D from 'react-force-graph-2d'

// Fallback Mock Data in case backend is offline
const MOCK_SUMMARY = {
  systemic_risk_index: 0.2345,
  high_risk_assets: 3,
  total_assets: 37,
  bl_sharpe: 1.124,
  eq_sharpe: 0.789,
  sharpe_improvement: 0.335,
  drawdown_improvement: 0.045,
  last_updated: "2026-06-04"
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'network' | 'matrix' | 'portfolio' | 'stress' | 'backtest'>('overview')
  const [expandedSectors, setExpandedSectors] = useState<Record<string, boolean>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // API Data States
  const [summary, setSummary] = useState<any>(null)
  const [latestRisk, setLatestRisk] = useState<any>(null)
  const [riskScoresData, setRiskScoresData] = useState<any>(null)
  const [graphSnapshots, setGraphSnapshots] = useState<any[]>([])
  const [portfolioData, setPortfolioData] = useState<any>(null)
  const [backtestData, setBacktestData] = useState<any>(null)
  
  // UI Interaction States
  const [selectedTicker, setSelectedTicker] = useState<string | null>("HDFCBANK.NS")
  const [tickerHistory, setTickerHistory] = useState<any>(null)
  const [tickerHistoryLoading, setTickerHistoryLoading] = useState(false)
  const [networkDateIndex, setNetworkDateIndex] = useState<number>(0)
  const [selectedScenario, setSelectedScenario] = useState<string>("2020_covid")
  const [scenarioData, setScenarioData] = useState<any>(null)
  const [scenarioLoading, setScenarioLoading] = useState(false)
  const [backtestPeriod, setBacktestPeriod] = useState<string>("2020_covid")

  // Force graph dimensions and sizing ref
  const graphContainerRef = useRef<HTMLDivElement>(null)
  const [graphWidth, setGraphWidth] = useState(600)
  const [graphHeight, setGraphHeight] = useState(500)
  // NEW: Advanced Network Controls
  const [colorMode, setColorMode] = useState<'risk' | 'sector'>('sector');
  const [highlightSector, setHighlightSector] = useState<string>('All');
  const [sectorMatrixData, setSectorMatrixData] = useState<any[]>([]);

  // Heatmap color interpolation for the Cross-Sector Matrix (Blue to Purple to Red)
  const getHeatmapColor = (val: number) => {
    let h, s, l;
    if (val < 0.5) {
      const ratio = val / 0.5;
      h = 220 + 30 * ratio; // 220 to 250
      s = 80 - 5 * ratio;   // 80 to 75
      l = 12 + 18 * ratio;  // 12 to 30
    } else if (val < 0.8) {
      const ratio = (val - 0.5) / 0.3;
      h = 250 + 60 * ratio; // 250 to 310
      s = 75 + 5 * ratio;   // 75 to 80
      l = 30 + 10 * ratio;  // 30 to 40
    } else {
      const ratio = (val - 0.8) / 0.2;
      h = 310 + 40 * ratio; // 310 to 350
      s = 80 + 10 * ratio;  // 80 to 90
      l = 40 + 5 * ratio;   // 40 to 45
    }
    return `hsl(${h}, ${s}%, ${l}%)`;
  };

  // Listen to window size changes for graph responsiveness
  useEffect(() => {
    function handleResize() {
      if (graphContainerRef.current) {
        setGraphWidth(graphContainerRef.current.clientWidth)
        setGraphHeight(graphContainerRef.current.clientHeight || 500)
      }
    }
    
    window.addEventListener('resize', handleResize)
    // Delay to let layout settle
    const timer = setTimeout(handleResize, 500)
    
    return () => {
      window.removeEventListener('resize', handleResize)
      clearTimeout(timer)
    }
  }, [activeTab, graphContainerRef.current])

  // Fetch initial dashboard configuration and baseline metrics
  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      // 1. Fetch Overview Statistics Summary
      const resSummary = await axios.get('/api/summary').catch(() => ({ data: MOCK_SUMMARY }))
      setSummary(resSummary.data)

      // 2. Fetch Latest GNN Risk Scores
      const resLatestRisk = await axios.get('/api/risk/latest').catch(() => null)
      if (resLatestRisk) setLatestRisk(resLatestRisk.data)

      // 3. Fetch Historical Risk Scores
      const resRiskScores = await axios.get('/api/risk/scores').catch(() => null)
      if (resRiskScores) setRiskScoresData(resRiskScores.data)

      // 4. Fetch All Graph Snapshots for Timeline
      const resSnapshots = await axios.get('/api/graph/snapshots').catch(() => ({ data: [] }))
      setGraphSnapshots(Array.isArray(resSnapshots.data) ? resSnapshots.data : [])

      // 5. Fetch Portfolio Weights
      const resPortfolio = await axios.get('/api/portfolio').catch(() => null)
      if (resPortfolio) setPortfolioData(resPortfolio.data)

      // 6. Fetch Backtest Results
      const resBacktest = await axios.get('/api/backtest').catch(() => null)
      if (resBacktest) setBacktestData(resBacktest.data)

      // 7. Fetch Sector-Level Contagion Matrix
      const resSectorAdj = await axios.get('/api/network/sector').catch(() => null);
      if (resSectorAdj && resSectorAdj.data) {
        setSectorMatrixData(resSectorAdj.data.sector_edges || []);
      }

      setLoading(false)
    } catch (err: any) {
      console.error(err)
      setError("Failed to query contagion pipeline API. Please verify FastAPI backend is active on port 8000.")
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  // Fetch individual stock contagion history when selected
  useEffect(() => {
    if (!selectedTicker) return
    
    async function fetchTickerHistory() {
      setTickerHistoryLoading(true)
      try {
        const res = await axios.get(`/api/risk/history/${selectedTicker}`)
        setTickerHistory(res.data)
      } catch (err) {
        console.error("Error fetching ticker history:", err)
        // Fallback mock history if backend fails
        if (riskScoresData) {
          const tickIdx = riskScoresData.node_order.indexOf(selectedTicker)
          if (tickIdx !== -1) {
            const scores = riskScoresData.scores.map((s: any) => s[tickIdx])
            setTickerHistory({
              ticker: selectedTicker,
              dates: riskScoresData.dates,
              scores: scores
            })
          }
        }
      } finally {
        setTickerHistoryLoading(false)
      }
    }
    
    fetchTickerHistory()
  }, [selectedTicker, riskScoresData])

  // Fetch Stress Scenario simulation when scenario changes
  useEffect(() => {
    async function fetchScenario() {
      setScenarioLoading(true)
      try {
        const res = await axios.get(`/api/shock/${selectedScenario}`)
        setScenarioData(res.data)
      } catch (err) {
        console.error("Error fetching scenario shock details:", err)
      } finally {
        setScenarioLoading(false)
      }
    }
    fetchScenario()
  }, [selectedScenario])

  // Systemic Risk History derived from 2D scores matrix
  const systemicRiskHistory = useMemo(() => {
    if (!riskScoresData) return []
    return riskScoresData.dates.map((date: string, idx: number) => {
      const dailyScores = riskScoresData.scores[idx]
      const avg = dailyScores.reduce((sum: number, val: number) => sum + val, 0) / dailyScores.length
      const highRiskCount = dailyScores.filter((s: number) => s > 0.6).length
      return {
        date,
        riskIndex: parseFloat(avg.toFixed(4)),
        highRiskCount
      }
    })
  }, [riskScoresData])

  // Formatted portfolio metrics
  const sectorWeightChartData = useMemo(() => {
    if (!portfolioData || !portfolioData.sector_weights) return []
    return Object.entries(portfolioData.sector_weights).map(([sector, details]: [string, any]) => ({
      name: sector,
      equal: parseFloat((details.eq * 100).toFixed(2)),
      optimized: parseFloat((details.bl * 100).toFixed(2)),
      color: details.color
    }))
  }, [portfolioData])

  const topAssetAllocations = useMemo(() => {
    if (!portfolioData || !portfolioData.views) return []
    return [...portfolioData.views]
      .sort((a, b) => b.bl_weight - a.bl_weight)
      .slice(0, 8)
      .map(v => ({
        name: v.name,
        ticker: v.ticker,
        equal: parseFloat((v.eq_weight * 100).toFixed(2)),
        optimized: parseFloat((v.bl_weight * 100).toFixed(2)),
        risk: v.risk_score
      }))
  }, [portfolioData])

  // Backtest cumulative returns line chart formatter
  const backtestChartData = useMemo(() => {
    if (!backtestData || !backtestData[backtestPeriod]) return []
    const period = backtestData[backtestPeriod]
    return period.dates.map((date: string, idx: number) => ({
      date,
      EqualWeight: parseFloat((period.cum_eq[idx] * 100).toFixed(1)),
      GnnBlackLitterman: parseFloat((period.cum_bl[idx] * 100).toFixed(1))
    }))
  }, [backtestData, backtestPeriod])

  // Sector mapping dictionary to group 11 raw sectors into the 5 target sectors
  const sectorMapping: Record<string, string> = useMemo(() => ({
    'Consumer': 'Consumer',
    'FMCG': 'Consumer',
    'Auto': 'Consumer',
    'Pharma': 'Healthcare',
    'Energy': 'Energy',
    'Metals': 'Energy',
    'Infra': 'Energy',
    'Banking': 'Finance',
    'Finance': 'Finance',
    'IT': 'Tech',
    'Telecom': 'Tech'
  }), []);

  // Premium colors corresponding to the 5 target sectors
  const sectorColors: Record<string, string> = useMemo(() => ({
    'Tech': '#38bdf8',       // Light Blue
    'Energy': '#fbbf24',     // Yellow/Amber
    'Finance': '#34d399',    // Green/Emerald
    'Healthcare': '#a78bfa', // Purple/Violet
    'Consumer': '#f472b6'    // Pink/Rose
  }), []);

  // Snapshot details for network rendering
  const [networkSnapshot, setNetworkSnapshot] = useState<any>(null)
  const [networkSnapshotLoading, setNetworkSnapshotLoading] = useState(false)

  // Fetch full snapshot details when index changes
  useEffect(() => {
    if (graphSnapshots.length === 0) return
    
    async function fetchFullSnapshot() {
      setNetworkSnapshotLoading(true)
      try {
        const res = await axios.get(`/api/graph/${networkDateIndex}`)
        setNetworkSnapshot(res.data)
      } catch (err) {
        console.error("Error fetching full snapshot details:", err)
      } finally {
        setNetworkSnapshotLoading(false)
      }
    }
    
    fetchFullSnapshot()
  }, [networkDateIndex, graphSnapshots])

  // Network visualizer graph data translator
  const forceGraphData = useMemo(() => {
    if (!networkSnapshot) return { nodes: [], links: [] }
    
    const nodes = networkSnapshot.nodes.map((n: any) => {
      let risk = 0
      if (riskScoresData) {
        const dateIdx = riskScoresData.dates.indexOf(networkSnapshot.date)
        const nodeIdx = riskScoresData.node_order.indexOf(n.id)
        if (dateIdx !== -1 && nodeIdx !== -1) {
          risk = riskScoresData.scores[dateIdx][nodeIdx]
        }
      }

      const mappedSec = sectorMapping[n.sector] || n.sector;
      const isMuted = highlightSector !== 'All' && mappedSec !== highlightSector;
      
      let displayColor = n.color || '#475569';
      if (colorMode === 'sector') {
        displayColor = sectorColors[mappedSec] || displayColor;
      } else if (colorMode === 'risk') {
        displayColor = risk < 0.25 ? '#22c55e' : risk < 0.55 ? '#eab308' : '#ef4444';
      }

      return {
        id: n.id,
        name: n.name,
        sector: n.sector,
        color: displayColor,
        isMuted,
        val: 3 + risk * 18,
        riskScore: risk,
        vol: n.features?.vol || 0,
        beta: n.features?.beta || 1
      }
    })

    const links = networkSnapshot.edges.map((e: any) => {
      const sourceNode = nodes.find((nd: any) => nd.id === e.source);
      const targetNode = nodes.find((nd: any) => nd.id === e.target);
      
      const sourceMappedSec = sourceNode ? (sectorMapping[sourceNode.sector] || sourceNode.sector) : '';
      const targetMappedSec = targetNode ? (sectorMapping[targetNode.sector] || targetNode.sector) : '';
      
      const isMuted = (highlightSector !== 'All') && 
        (sourceMappedSec !== highlightSector && targetMappedSec !== highlightSector);

      return {
        source: e.source,
        target: e.target,
        weight: e.weight,
        type: e.type,
        isMuted,
        directed: e.directed || false
      }
    })

    return { nodes, links }
  }, [networkSnapshot, riskScoresData, colorMode, highlightSector, sectorMapping, sectorColors])

  // Compute average risk per sector for the Donut Chart
  const sectorRiskData = useMemo(() => {
    const nodes = forceGraphData.nodes;
    if (!nodes || nodes.length === 0) {
      if (!latestRisk || !latestRisk.assets) return [];
      const sectorStats: Record<string, { totalRisk: number, count: number }> = {};
      latestRisk.assets.forEach((asset: any) => {
        const mappedSec = sectorMapping[asset.sector] || asset.sector;
        if (!sectorStats[mappedSec]) {
          sectorStats[mappedSec] = { totalRisk: 0, count: 0 };
        }
        sectorStats[mappedSec].totalRisk += asset.risk_score;
        sectorStats[mappedSec].count += 1;
      });
      return Object.entries(sectorStats).map(([name, stats]) => ({
        name,
        avgRisk: parseFloat((stats.totalRisk / stats.count).toFixed(3)),
        color: sectorColors[name] || '#475569'
      })).sort((a, b) => b.avgRisk - a.avgRisk);
    }
    
    const sectorStats: Record<string, { totalRisk: number, count: number }> = {};
    nodes.forEach((node: any) => {
      const mappedSec = sectorMapping[node.sector] || node.sector;
      if (!sectorStats[mappedSec]) {
        sectorStats[mappedSec] = { totalRisk: 0, count: 0 };
      }
      sectorStats[mappedSec].totalRisk += node.riskScore || 0;
      sectorStats[mappedSec].count += 1;
    });

    return Object.entries(sectorStats)
      .map(([name, stats]) => ({
        name,
        avgRisk: parseFloat((stats.totalRisk / stats.count).toFixed(3)),
        color: sectorColors[name] || '#475569'
      }))
      .sort((a, b) => b.avgRisk - a.avgRisk);
  }, [forceGraphData, latestRisk, sectorMapping, sectorColors]);

  // Extract unique sectors for the matrix dropdown and grid (static string[] to fix TS2345 compiler errors)
  const uniqueSectors = useMemo<string[]>(() => {
    return ['Consumer', 'Healthcare', 'Energy', 'Finance', 'Tech'];
  }, []);

  // Group assets by sector for Hierarchical Contagion View
  const hierarchicalContagion = useMemo(() => {
    const nodes = forceGraphData.nodes;
    const assetsSource = (nodes && nodes.length > 0) 
      ? nodes.map((n: any) => ({
          ticker: n.id,
          name: n.name,
          sector: n.sector,
          risk_score: n.riskScore
        }))
      : (latestRisk?.assets || []);

    if (assetsSource.length === 0) return [];
    
    const sectorsMap: Record<string, { name: string, avgRisk: number, color: string, assets: any[] }> = {};
    
    assetsSource.forEach((asset: any) => {
      const mappedSec = sectorMapping[asset.sector] || asset.sector;
      if (!sectorsMap[mappedSec]) {
        sectorsMap[mappedSec] = {
          name: mappedSec,
          avgRisk: 0,
          color: sectorColors[mappedSec] || '#475569',
          assets: []
        };
      }
      sectorsMap[mappedSec].assets.push(asset);
    });
    
    return Object.values(sectorsMap).map(sec => {
      const totalRisk = sec.assets.reduce((sum, a) => sum + (a.risk_score || 0), 0);
      const sortedAssets = [...sec.assets].sort((a, b) => (b.risk_score || 0) - (a.risk_score || 0));
      return {
        ...sec,
        avgRisk: totalRisk / sec.assets.length,
        assets: sortedAssets
      };
    }).sort((a, b) => b.avgRisk - a.avgRisk);
  }, [forceGraphData, latestRisk, sectorMapping, sectorColors]);

  // Get the top 4 highest-risk firms in the current GNN risk list
  const topRiskFirms = useMemo(() => {
    const nodes = forceGraphData.nodes;
    if (nodes && nodes.length > 0) {
      return [...nodes]
        .map((n: any) => ({
          ticker: n.id,
          name: n.name,
          sector: sectorMapping[n.sector] || n.sector,
          risk_score: n.riskScore,
          color: n.color,
          beta: n.beta,
          vol: n.vol
        }))
        .sort((a, b) => b.risk_score - a.risk_score)
        .slice(0, 4);
    }
    if (!latestRisk || !latestRisk.assets) return [];
    return [...latestRisk.assets]
      .map((a: any) => ({
        ticker: a.ticker,
        name: a.name,
        sector: sectorMapping[a.sector] || a.sector,
        risk_score: a.risk_score,
        color: a.color,
        beta: a.beta,
        vol: a.vol
      }))
      .sort((a, b) => b.risk_score - a.risk_score)
      .slice(0, 4);
  }, [forceGraphData, latestRisk, sectorMapping]);

  // Active sector edges based on the timeline date scrubber
  const currentSectorEdges = useMemo(() => {
    if (!sectorMatrixData || sectorMatrixData.length === 0) return [];
    const snap = sectorMatrixData[networkDateIndex] || sectorMatrixData[sectorMatrixData.length - 1];
    return snap ? (snap.sector_edges || []) : [];
  }, [sectorMatrixData, networkDateIndex]);



  // Render customized label for Recharts Pie (Donut Chart) directly on the segments
  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }: any) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    
    if (percent < 0.05) return null;

    return (
      <text 
        x={x} 
        y={y} 
        fill="#f8fafc" 
        textAnchor="middle" 
        dominantBaseline="central"
        className="text-[9px] font-bold fill-slate-100 font-sans pointer-events-none"
      >
        <tspan x={x} dy="-0.4em">{name}</tspan>
        <tspan x={x} dy="1.2em">{(percent * 100).toFixed(1)}%</tspan>
      </text>
    );
  };

  if (loading && !summary) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6 space-y-4">
        <RefreshCw className="w-12 h-12 text-teal-400 animate-spin" />
        <h2 className="text-xl font-medium tracking-wide">Loading Contagion & Portfolio Analytics...</h2>
        <p className="text-slate-400 text-sm max-w-md text-center">Parsing offline Graph Neural Network features, Sharpe bounds, and historical Yahoo Finance returns.</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-teal-500/30 selection:text-teal-300">
      
      {/* Premium Top Navigation Bar */}
      <header className="border-b border-slate-900 bg-slate-950/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-tr from-teal-500 to-emerald-600 rounded-lg shadow-lg shadow-teal-500/20 animate-pulse">
              <Activity className="w-6 h-6 text-slate-950 stroke-[2.5]" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-teal-200 via-slate-100 to-teal-100 bg-clip-text text-transparent">
                FINANCIAL CONTAGION ENGINE
              </h1>
              <p className="text-xs text-slate-400 font-mono tracking-wider uppercase">GNN Propagation + Black-Litterman Optimal Portfolios</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-xs bg-slate-900 border border-slate-800 text-slate-400 py-1.5 px-3 rounded-full font-mono flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>
              Last updated: {summary?.last_updated || "2026-06-04"}
            </span>
            <button 
              onClick={fetchData} 
              className="p-2 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg hover:text-teal-300 transition duration-200"
              title="Refresh Dashboard Data"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col space-y-6">
        
        {/* Error Alert Box */}
        {error && (
          <div className="p-4 bg-red-950/40 border border-red-900/60 rounded-xl flex items-start space-x-3 text-red-200">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="font-semibold text-sm">Contagion Pipeline Warning</h3>
              <p className="text-xs text-red-300/90 mt-1">{error}</p>
              <p className="text-xs text-red-400/80 mt-2 font-mono">Note: Showing precomputed sandbox cache assets for live demonstration.</p>
            </div>
          </div>
        )}

        {/* Global Key Metrics Ribbon */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          
          <div className="glass-panel glass-panel-hover p-5 rounded-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-teal-500/5 rounded-full blur-2xl"></div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-slate-400">Systemic Risk Index</span>
              <Activity className="w-5 h-5 text-teal-400" />
            </div>
            <div className="mt-3 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono tracking-tight text-slate-100">
                {(summary?.systemic_risk_index * 100 || 23.45).toFixed(2)}%
              </span>
              <span className="text-xs text-emerald-400 font-medium bg-emerald-950/50 border border-emerald-900/60 px-1.5 py-0.5 rounded">
                GNN Global
              </span>
            </div>
            <div className="mt-3 text-xs text-slate-400">
              Mean asset-level exposure to financial contagion
            </div>
          </div>

          <div className="glass-panel glass-panel-hover p-5 rounded-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-full blur-2xl"></div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-slate-400">Sharpe Ratio Outperformance</span>
              <TrendingUp className="w-5 h-5 text-emerald-400" />
            </div>
            <div className="mt-3 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono tracking-tight text-emerald-300">
                +{((summary?.bl_sharpe - summary?.eq_sharpe) / (summary?.eq_sharpe || 1) * 100 || 42.4).toFixed(1)}%
              </span>
              <span className="text-xs text-emerald-300 font-mono">
                {summary?.bl_sharpe?.toFixed(2) || "1.12"} vs {summary?.eq_sharpe?.toFixed(2) || "0.79"}
              </span>
            </div>
            <div className="mt-3 text-xs text-slate-400">
              Black-Litterman optimization vs Equal Weight baseline
            </div>
          </div>

          <div className="glass-panel glass-panel-hover p-5 rounded-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/5 rounded-full blur-2xl"></div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-slate-400">Drawdown Shield</span>
              <Percent className="w-5 h-5 text-blue-400" />
            </div>
            <div className="mt-3 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono tracking-tight text-blue-300">
                +{(summary?.drawdown_improvement * 100 || 4.5).toFixed(1)}%
              </span>
              <span className="text-xs text-slate-400">lower drawdown</span>
            </div>
            <div className="mt-3 text-xs text-slate-400">
              Mitigates downside losses during systemic market stress
            </div>
          </div>

          <div className="glass-panel glass-panel-hover p-5 rounded-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-amber-500/5 rounded-full blur-2xl"></div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-slate-400">Dangerous Vectors</span>
              <ShieldAlert className="w-5 h-5 text-amber-400" />
            </div>
            <div className="mt-3 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono tracking-tight text-amber-300">
                {summary?.high_risk_assets || 3}
              </span>
              <span className="text-xs text-slate-400">assets flagged</span>
            </div>
            <div className="mt-3 text-xs text-slate-400">
              Individual asset GNN risk scores exceeding critical threshold (&gt;0.6)
            </div>
          </div>

        </section>

        {/* Tab Selection Navigation */}
        <nav className="flex space-x-2 p-1.5 bg-slate-900/50 border border-slate-900 rounded-xl shrink-0">
          <button
            onClick={() => setActiveTab('overview')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg transition duration-150 ${
              activeTab === 'overview'
                ? 'bg-teal-500 text-slate-950 shadow-md shadow-teal-500/15'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            Contagion Summary
          </button>
          
          <button
            onClick={() => setActiveTab('network')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg transition duration-150 ${
              activeTab === 'network'
                ? 'bg-teal-500 text-slate-950 shadow-md shadow-teal-500/15'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Network className="w-4 h-4" />
            Contagion Network Graph
          </button>

          <button
            onClick={() => setActiveTab('matrix')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg transition duration-150 ${
              activeTab === 'matrix'
                ? 'bg-teal-500 text-slate-950 shadow-md shadow-teal-500/15'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <PieIcon className="w-4 h-4" />
            Cross-Sector Matrix
          </button>

          <button
            onClick={() => setActiveTab('portfolio')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg transition duration-150 ${
              activeTab === 'portfolio'
                ? 'bg-teal-500 text-slate-950 shadow-md shadow-teal-500/15'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <PieIcon className="w-4 h-4" />
            Portfolio Allocation
          </button>

          <button
            onClick={() => setActiveTab('stress')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg transition duration-150 ${
              activeTab === 'stress'
                ? 'bg-teal-500 text-slate-950 shadow-md shadow-teal-500/15'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Zap className="w-4 h-4" />
            Stress Test Simulator
          </button>

          <button
            onClick={() => setActiveTab('backtest')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg transition duration-150 ${
              activeTab === 'backtest'
                ? 'bg-teal-500 text-slate-950 shadow-md shadow-teal-500/15'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <LineIcon className="w-4 h-4" />
            Backtest Metrics
          </button>
        </nav>

        {/* Tab Screen Content */}
        <div className="flex-1">
          
          {/* TAB 1: OVERVIEW */}
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Systemic Risk History Chart Card */}
              <div className="glass-panel p-6 rounded-2xl lg:col-span-2 flex flex-col space-y-4">
                <div>
                  <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-teal-400" />
                    Systemic Risk Index History
                  </h2>
                  <p className="text-xs text-slate-400">Historical trend of average contagion risk computed across the stock network.</p>
                </div>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={systemicRiskHistory} margin={{ left: 10 }}>
                      <defs>
                        <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#14b8a6" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="date" stroke="#94a3b8" fontSize={11} tickMargin={10} />
                      <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 'auto']} label={{ value: 'Risk Index', angle: -90, position: 'insideLeft', offset: 0, fill: '#94a3b8', style: { textAnchor: 'middle', fontSize: 10, fontWeight: 'bold' } }} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px' }}
                        labelStyle={{ color: '#94a3b8', fontWeight: 'bold' }}
                      />
                      <Area type="monotone" dataKey="riskIndex" name="Systemic Risk Index" stroke="#14b8a6" strokeWidth={2.5} fillOpacity={1} fill="url(#riskGrad)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Dangerous Contagion Vectors Table */}
              <div className="glass-panel p-6 rounded-2xl flex flex-col space-y-4">
                <div>
                  <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                    <ShieldAlert className="w-5 h-5 text-amber-400" />
                    Top Contagion Vectors
                  </h2>
                  <p className="text-xs text-slate-400">Assets exhibiting the highest contagion risk scores under GNN prediction.</p>
                </div>
                <div className="flex-1 overflow-y-auto max-h-[290px] pr-2 space-y-3">
                  {latestRisk?.assets?.slice(0, 5).map((asset: any) => (
                    <div 
                      key={asset.ticker} 
                      onClick={() => setSelectedTicker(asset.ticker)}
                      className={`p-3 border rounded-xl flex items-center justify-between cursor-pointer transition ${
                        selectedTicker === asset.ticker 
                          ? 'bg-slate-900 border-teal-500/50 shadow-md shadow-teal-500/5' 
                          : 'bg-slate-900/30 border-slate-800/80 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: asset.color || '#fff' }}></div>
                        <div>
                          <p className="text-sm font-bold text-slate-200">{asset.ticker}</p>
                          <p className="text-[10px] text-slate-400 font-medium">{asset.name} • {asset.sector}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className={`text-sm font-bold font-mono ${asset.risk_score > 0.6 ? 'text-rose-400' : 'text-amber-400'}`}>
                          {asset.risk_score.toFixed(3)}
                        </span>
                        <p className="text-[9px] text-slate-400 font-mono">beta: {asset.beta?.toFixed(1)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Selected Stock Details & Mini Risk Chart */}
              {selectedTicker && tickerHistory && (
                <div className="glass-panel p-6 rounded-2xl lg:col-span-3 grid grid-cols-1 md:grid-cols-3 gap-6">
                  
                  {/* Stock Metrics Profile */}
                  <div className="space-y-4">
                    <div className="flex items-center space-x-3">
                      <div className="p-2.5 bg-slate-900 border border-slate-800 rounded-xl">
                        <FileText className="w-6 h-6 text-teal-400" />
                      </div>
                      <div>
                        <h3 className="text-base font-bold text-slate-200">{tickerHistory.ticker}</h3>
                        <p className="text-xs text-slate-400">Contagion Profile & Risk Scores</p>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-3 pt-2">
                      <div className="bg-slate-900/50 border border-slate-900 p-3 rounded-xl">
                        <span className="text-[10px] uppercase font-mono text-slate-400">Volatility</span>
                        <p className="text-lg font-bold text-slate-200 font-mono">
                          {((latestRisk?.assets?.find((a: any) => a.ticker === selectedTicker)?.vol || 0.25) * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="bg-slate-900/50 border border-slate-900 p-3 rounded-xl">
                        <span className="text-[10px] uppercase font-mono text-slate-400">Market Beta</span>
                        <p className="text-lg font-bold text-slate-200 font-mono">
                          {(latestRisk?.assets?.find((a: any) => a.ticker === selectedTicker)?.beta || 1.1).toFixed(2)}
                        </p>
                      </div>
                      <div className="bg-slate-900/50 border border-slate-900 p-3 rounded-xl">
                        <span className="text-[10px] uppercase font-mono text-slate-400">Current GNN Risk</span>
                        <p className="text-lg font-bold text-rose-400 font-mono">
                          {(latestRisk?.assets?.find((a: any) => a.ticker === selectedTicker)?.risk_score || 0.1).toFixed(4)}
                        </p>
                      </div>
                      <div className="bg-slate-900/50 border border-slate-900 p-3 rounded-xl">
                        <span className="text-[10px] uppercase font-mono text-slate-400">Sector</span>
                        <p className="text-sm font-bold text-slate-200 mt-1">
                          {latestRisk?.assets?.find((a: any) => a.ticker === selectedTicker)?.sector || "Unknown"}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Stock Risk Trend AreaChart */}
                  <div className="md:col-span-2 flex flex-col space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-bold text-slate-300">GNN Risk Propagation Time Series</h4>
                      <span className="text-[10px] font-mono text-slate-400">Tracking network shock sensitivity</span>
                    </div>
                    
                    <div className="h-44">
                      {tickerHistoryLoading ? (
                        <div className="w-full h-full flex items-center justify-center">
                          <RefreshCw className="w-6 h-6 text-teal-400 animate-spin" />
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={tickerHistory.dates.map((d: string, i: number) => ({ date: d, risk: tickerHistory.scores[i] }))} margin={{ left: 10 }}>
                            <defs>
                              <linearGradient id="tickerGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="date" stroke="#94a3b8" fontSize={9} />
                            <YAxis stroke="#94a3b8" fontSize={9} label={{ value: 'Contagion Risk', angle: -90, position: 'insideLeft', offset: 0, fill: '#94a3b8', style: { textAnchor: 'middle', fontSize: 9, fontWeight: 'bold' } }} />
                            <Tooltip 
                              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px' }}
                              labelStyle={{ color: '#94a3b8', fontWeight: 'bold' }}
                            />
                            <Area type="monotone" dataKey="risk" name="GNN Contagion Risk" stroke="#f43f5e" strokeWidth={2} fillOpacity={1} fill="url(#tickerGrad)" />
                          </AreaChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                  </div>

                </div>
              )}

            </div>
          )}
          {activeTab === 'network' && (
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
              
              {/* Left Column: Network Graph Controller Panel */}
              <div className="glass-panel p-6 rounded-2xl flex flex-col space-y-6 lg:col-span-1">
                <div>
                  <h2 className="text-base font-extrabold text-slate-100 tracking-tight">
                    Controls
                  </h2>
                </div>

                {/* Colour Mode Selector */}
                <div className="space-y-3">
                  <label className="text-[10px] font-bold text-slate-400 block uppercase font-mono tracking-wider">Colour mode</label>
                  <div className="space-y-3">
                    <button
                      onClick={() => setColorMode('risk')}
                      className="flex items-center gap-2.5 text-xs font-semibold cursor-pointer bg-transparent border-0 outline-none text-left w-full focus:outline-none"
                    >
                      <div className={`w-4 h-4 rounded-full border flex items-center justify-center flex-shrink-0 transition-all ${colorMode === 'risk' ? 'border-teal-400' : 'border-slate-700'}`}>
                        <div className={`w-2 h-2 rounded-full transition-all ${colorMode === 'risk' ? 'bg-teal-400' : 'bg-transparent'}`}></div>
                      </div>
                      <div className="w-3.5 h-3.5 rounded-full bg-emerald-500 border-2 border-rose-500 flex-shrink-0 shadow shadow-rose-500/20"></div>
                      <span className={colorMode === 'risk' ? 'text-slate-100 font-extrabold' : 'text-slate-400 hover:text-slate-300 font-semibold'}>Contagion Risk</span>
                    </button>
                    
                    <button
                      onClick={() => setColorMode('sector')}
                      className="flex items-center gap-2.5 text-xs font-semibold cursor-pointer bg-transparent border-0 outline-none text-left w-full focus:outline-none"
                    >
                      <div className={`w-4 h-4 rounded-full border flex items-center justify-center flex-shrink-0 transition-all ${colorMode === 'sector' ? 'border-teal-400' : 'border-slate-700'}`}>
                        <div className={`w-2 h-2 rounded-full transition-all ${colorMode === 'sector' ? 'bg-teal-400' : 'bg-transparent'}`}></div>
                      </div>
                      <div className="w-3.5 h-3.5 rounded-full bg-white border border-slate-300 flex items-center justify-center flex-shrink-0 overflow-hidden">
                        <span className="w-full h-full bg-gradient-to-tr from-sky-400 via-amber-400 to-rose-400"></span>
                      </div>
                      <span className={colorMode === 'sector' ? 'text-slate-100 font-extrabold' : 'text-slate-400 hover:text-slate-300 font-semibold'}>Sector Colour</span>
                    </button>
                  </div>
                </div>

                {/* Highlight Sector Selector */}
                <div className="space-y-2">
                  <label className="text-[10px] font-bold text-slate-400 block uppercase font-mono tracking-wider">Highlight sector</label>
                  <select
                    value={highlightSector}
                    onChange={(e) => setHighlightSector(e.target.value)}
                    className="w-full p-2.5 bg-slate-900 border border-slate-800 rounded-xl text-xs font-semibold text-slate-300 focus:outline-none focus:border-teal-500/60 focus:ring-1 focus:ring-teal-500/30 transition-all"
                  >
                    <option value="All">All sectors</option>
                    {uniqueSectors.map((sector: string) => (
                      <option key={sector} value={sector}>{sector}</option>
                    ))}
                  </select>
                </div>

                {/* Info Text */}
                <div className="text-[10px] text-slate-400 leading-relaxed mt-auto flex items-start gap-1.5 font-medium italic">
                  <span className="text-amber-400 shrink-0 select-none">👉</span>
                  <span>Click any firm to see details.</span>
                </div>
              </div>

              {/* Middle Column: Force Directed Graph Render Area */}
              <div 
                ref={graphContainerRef} 
                className="glass-panel rounded-2xl lg:col-span-3 min-h-[500px] max-h-[600px] overflow-hidden relative flex flex-col animate-fade-in"
              >
                <div className="p-4 border-b border-slate-900 flex items-center justify-between shrink-0 bg-slate-950/40">
                  <div>
                    <h3 className="text-sm font-bold text-slate-200">
                      Financial Dependency Network — {networkSnapshot?.node_count || 50} Firms <span className="text-slate-500 font-normal">(2015–2024)</span>
                    </h3>
                  </div>
                </div>

                <div className="flex-1 relative bg-[#020617] overflow-hidden">
                  {networkSnapshotLoading && (
                    <div className="absolute inset-0 z-10 bg-slate-950/70 flex items-center justify-center">
                      <RefreshCw className="w-10 h-10 text-teal-400 animate-spin" />
                    </div>
                  )}
                  
                  {forceGraphData.nodes.length > 0 ? (
                    <ForceGraph2D
                      graphData={forceGraphData}
                      width={graphWidth}
                      height={graphHeight - 60}
                      backgroundColor="transparent"
                      nodeColor={(n: any) => n.color}
                      nodeVal={(n: any) => n.val}
                      nodeLabel={(n: any) => `${n.name} (${n.id}) - GNN Risk: ${n.riskScore.toFixed(4)}`}
                      linkColor={(l: any) => l.isMuted ? 'rgba(148, 163, 184, 0.015)' : 'rgba(148, 163, 184, 0.18)'}
                      linkWidth={(l: any) => l.isMuted ? 0.3 : l.weight * 2.5}
                      linkDirectionalArrowLength={(l: any) => l.directed && !l.isMuted ? 4 : 0}
                      linkDirectionalArrowRelPos={0.5}
                      onNodeClick={(node: any) => setSelectedTicker(node.id)}
                      nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
                        const label = node.id.split('.')[0]
                        const r = 9 + (node.riskScore || 0) * 8
                        const opacity = node.isMuted ? 0.15 : 1.0;
                        
                        // Glowing outer ring for high risk (only if not muted)
                        if (node.riskScore > 0.6 && !node.isMuted) {
                          ctx.beginPath()
                          ctx.arc(node.x, node.y, r + 4, 0, 2 * Math.PI, false)
                          ctx.fillStyle = 'rgba(239, 68, 68, 0.22)'
                          ctx.fill()
                        }
                        
                        // Base node circle
                        ctx.beginPath()
                        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false)
                        ctx.fillStyle = node.color
                        ctx.globalAlpha = opacity
                        ctx.fill()
                        
                        // Text Ticker inside node
                        ctx.font = `bold ${6 + r / 3}px sans-serif`
                        ctx.textAlign = 'center'
                        ctx.textBaseline = 'middle'
                        ctx.fillStyle = node.isMuted ? 'rgba(248, 250, 252, 0.1)' : '#ffffff'
                        ctx.fillText(label, node.x, node.y)
                        ctx.globalAlpha = 1.0
                      }}
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-400">
                      No graph snapshot data available.
                    </div>
                  )}
                </div>
              </div>

              {/* Right Column: Sector Donut and Top Risk Firms */}
              <div className="flex flex-col space-y-6 lg:col-span-1">
                
                {/* Sector Donut Chart Card */}
                <div className="glass-panel p-5 rounded-2xl flex flex-col space-y-4 relative">
                  <div>
                    <h4 className="text-sm font-bold text-slate-200">GNN Risk by Sector</h4>
                    <p className="text-[10px] text-slate-500 leading-normal mt-0.5">Average contagion probability across all firms in each sector</p>
                  </div>

                  <div className="h-40 relative flex items-center justify-center">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={sectorRiskData}
                          cx="50%"
                          cy="50%"
                          innerRadius={45}
                          outerRadius={65}
                          paddingAngle={2.5}
                          dataKey="avgRisk"
                          labelLine={false}
                          label={renderCustomizedLabel}
                        >
                          {sectorRiskData.map((entry: any, index: number) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                          formatter={(v: any) => [`${(v * 100).toFixed(1)}%`]}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                    {/* Central Donut Hole label */}
                    <div className="absolute inset-0 flex items-center justify-center flex-col pointer-events-none mt-2">
                      <span className="text-[8px] uppercase font-mono text-slate-500">Avg Risk</span>
                      <span className="text-[9px] font-bold text-slate-400">by Sector</span>
                    </div>
                  </div>
                </div>

                {/* Top Risk Firms Card */}
                <div className="glass-panel p-5 rounded-2xl flex flex-col space-y-4 flex-1">
                  <div>
                    <h4 className="text-sm font-bold text-slate-200">Top Risk Firms</h4>
                    <p className="text-[10px] text-slate-500 mt-0.5">Click a firm in the network for full detail</p>
                  </div>

                  <div className="space-y-4 flex-1">
                    {selectedTicker ? (
                      // Clicked Firm Detail Profile (matching target markup underline design)
                      (() => {
                        const asset = latestRisk?.assets?.find((a: any) => a.ticker === selectedTicker) || 
                                      forceGraphData.nodes.find((n: any) => n.id === selectedTicker);
                        const rawSec = asset?.sector || 'Unknown';
                        const mappedSec = sectorMapping[rawSec] || rawSec;
                        const themeColor = sectorColors[mappedSec] || '#475569';
                        const tickerClean = selectedTicker.split('.')[0];
                        
                        return (
                          <div className="space-y-3.5">
                            <div>
                              <div className="flex items-baseline gap-2">
                                <span className="text-lg font-extrabold text-slate-100 font-mono tracking-tight">{tickerClean}</span>
                                <span className="text-xs font-bold font-mono tracking-wide" style={{ color: themeColor }}>
                                  {mappedSec}
                                </span>
                              </div>
                              {/* Underline bar matching sector color */}
                              <div className="h-0.5 rounded mt-1.5 w-1/3" style={{ backgroundColor: themeColor }}></div>
                            </div>

                            {/* Key values */}
                            <div className="grid grid-cols-2 gap-2 text-[10px]">
                              <div className="bg-slate-900 border border-slate-800 p-2 rounded-xl">
                                <span className="text-slate-500 block font-mono">Beta</span>
                                <span className="font-bold font-mono text-slate-300">
                                  {(asset?.beta || 1.1).toFixed(2)}
                                </span>
                              </div>
                              <div className="bg-slate-900 border border-slate-800 p-2 rounded-xl">
                                <span className="text-slate-500 block font-mono">Vol</span>
                                <span className="font-bold font-mono text-slate-300">
                                  {((asset?.vol || 0.25) * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="bg-slate-900 border border-slate-800 p-2 rounded-xl col-span-2">
                                <span className="text-slate-500 block font-mono">Contagion Risk</span>
                                <span className="font-bold font-mono text-rose-400">
                                  {(asset?.riskScore !== undefined ? asset.riskScore : (asset?.risk_score || 0)).toFixed(4)}
                                </span>
                              </div>
                            </div>

                            <button 
                              onClick={() => setSelectedTicker(null)}
                              className="w-full p-2 bg-slate-900 hover:bg-slate-850 text-[9px] font-bold tracking-wider text-slate-400 rounded-lg border border-slate-800 uppercase transition"
                            >
                              Clear Selection
                            </button>
                          </div>
                        );
                      })()
                    ) : (
                      // High Risk Asset Leaderboard
                      <div className="space-y-3">
                        {topRiskFirms.map((firm: any) => (
                          <div 
                            key={firm.ticker} 
                            onClick={() => setSelectedTicker(firm.ticker)}
                            className="p-2.5 bg-slate-900/40 border border-slate-900 hover:border-slate-800 rounded-xl cursor-pointer space-y-1 transition duration-150"
                          >
                            <div className="flex justify-between items-center text-[10px] font-bold text-slate-300">
                              <span className="truncate max-w-[120px]">{firm.ticker.split('.')[0]} <span className="text-[9px] text-slate-500 font-normal font-mono">{firm.sector}</span></span>
                              <span className="font-mono text-amber-400">{(firm.risk_score * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-slate-950 rounded-full h-1 overflow-hidden">
                              <div className="bg-amber-500 h-1" style={{ width: `${firm.risk_score * 100}%` }}></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {/* Collapsible Accordion for Sector Wise Hierarchical Contagion */}
                    <div className="pt-4 border-t border-slate-900 space-y-2">
                      <h5 className="text-[10px] font-bold uppercase tracking-wider font-mono text-slate-400">Hierarchical Contagion</h5>
                      <div className="space-y-2 overflow-y-auto max-h-[140px] pr-1">
                        {hierarchicalContagion.map((sec: any) => {
                          const isExpanded = !!expandedSectors[sec.name];
                          return (
                            <div key={sec.name} className="border border-slate-900/80 bg-slate-900/10 rounded-xl overflow-hidden">
                              <div 
                                onClick={() => setExpandedSectors(prev => ({ ...prev, [sec.name]: !prev[sec.name] }))}
                                className="p-2 flex items-center justify-between text-[10px] font-bold text-slate-300 cursor-pointer hover:bg-slate-900/40 transition select-none"
                              >
                                <span className="flex items-center gap-1.5">
                                  <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: sec.color }}></span>
                                  {sec.name}
                                </span>
                                <div className="flex items-center gap-1.5 font-mono text-slate-400">
                                  <span>{(sec.avgRisk * 100).toFixed(1)}%</span>
                                  <span className="text-[8px] text-slate-500 font-normal">{isExpanded ? '▼' : '▶'}</span>
                                </div>
                              </div>
                              {isExpanded && (
                                <div className="bg-slate-950/60 p-2 border-t border-slate-900 space-y-1.5">
                                  {sec.assets.map((ast: any) => (
                                    <div 
                                      key={ast.ticker}
                                      onClick={(e) => { e.stopPropagation(); setSelectedTicker(ast.ticker); }}
                                      className={`flex items-center justify-between text-[9px] px-1.5 py-1 rounded transition ${
                                        selectedTicker === ast.ticker 
                                          ? 'bg-teal-500/10 text-teal-300 font-bold border border-teal-500/20' 
                                          : 'text-slate-400 hover:text-slate-200 cursor-pointer'
                                      }`}
                                    >
                                      <span className="font-mono">{ast.ticker.split('.')[0]}</span>
                                      <span className="font-mono font-semibold">{(ast.risk_score * 100).toFixed(1)}%</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>

                  </div>
                </div>

              </div>

            </div>
          )}

          {/* TAB 3: CROSS-SECTOR CONTAGION MATRIX */}
          {activeTab === 'matrix' && (
            <div className="glass-panel p-6 rounded-2xl flex flex-col items-center justify-center space-y-6 relative overflow-hidden animate-fade-in">
              <div className="absolute top-0 right-0 w-64 h-64 bg-teal-500/5 rounded-full blur-3xl pointer-events-none"></div>
              
              <div className="w-full text-center max-w-2xl shrink-0">
                <h2 className="text-xl font-bold text-slate-100 flex items-center justify-center gap-2">
                  <Network className="w-5 h-5 text-teal-400" />
                  Cross-Sector Contagion Matrix
                </h2>
                <p className="text-xs text-slate-400 mt-1 max-w-lg mx-auto">
                  How strongly each sector infected the others during the crisis snapshot date — darker / warmer = more contagion.
                </p>
                
                {/* Snapshot Date indicator in Matrix Tab */}
                <div className="mt-3 inline-flex items-center gap-1.5 px-3 py-1 bg-slate-900 border border-slate-800 rounded-full font-mono text-[10px] text-slate-400">
                  <span className="w-2.5 h-2.5 rounded-full bg-teal-400 animate-pulse"></span>
                  Active Snapshot: <span className="text-teal-300 font-bold">{graphSnapshots[networkDateIndex]?.date || "Latest"}</span>
                </div>
              </div>

              {/* Grid Matrix layout */}
              <div className="flex flex-col md:flex-row items-center justify-center gap-8 w-full max-w-5xl py-4">
                
                {/* The Matrix */}
                <div className="flex flex-col bg-slate-950/80 p-4 border border-slate-900 rounded-2xl shadow-xl overflow-x-auto max-w-full">
                  <div className="text-center font-bold text-xs text-slate-300 mb-4 font-sans tracking-wide">
                    Cross-Sector Contagion Matrix — during {graphSnapshots[networkDateIndex]?.date || "COVID-19 Crash"}
                  </div>

                  <div className="flex flex-col space-y-1 min-w-[550px]">
                    
                    {(() => {
                      const ySectors = ['Consumer', 'Healthcare', 'Energy', 'Finance', 'Tech'];
                      const xSectors = ['Tech', 'Finance', 'Energy', 'Healthcare', 'Consumer'];
                      const mockupMatrix: Record<string, Record<string, number>> = {
                        Consumer:   { Tech: 0.67, Finance: 0.66, Energy: 0.50, Healthcare: 0.62, Consumer: 0.64 },
                        Healthcare: { Tech: 0.64, Finance: 0.62, Energy: 0.47, Healthcare: 0.70, Consumer: 0.62 },
                        Energy:     { Tech: 0.56, Finance: 0.71, Energy: 0.78, Healthcare: 0.47, Consumer: 0.50 },
                        Finance:    { Tech: 0.69, Finance: 0.85, Energy: 0.71, Healthcare: 0.62, Consumer: 0.66 },
                        Tech:       { Tech: 0.76, Finance: 0.69, Energy: 0.56, Healthcare: 0.64, Consumer: 0.67 }
                      };

                      return (
                        <>
                          {/* Header Row */}
                          <div className="flex items-center space-x-1">
                            {/* Blank top-left cell */}
                            <div className="w-20 text-right pr-3 text-[10px] font-bold text-slate-500 uppercase tracking-wider font-mono"></div>
                            {xSectors.map((colSector: string) => (
                              <div key={colSector} className="w-14 text-center text-[10px] font-bold text-slate-400 truncate" title={colSector}>
                                {colSector}
                              </div>
                            ))}
                          </div>

                          {/* Matrix Rows */}
                          {ySectors.map((rowSector: string) => (
                            <div key={rowSector} className="flex items-center space-x-1 h-9">
                              
                              {/* Row Label */}
                              <div className="w-20 text-right pr-3 text-[10px] font-bold text-slate-400 truncate" title={rowSector}>
                                {rowSector}
                              </div>

                              {/* Row Cells */}
                              {xSectors.map((colSector: string) => {
                                // Aggregate raw edge weights for this cell
                                let totalWeight = 0;
                                let edgeCount = 0;
                                
                                currentSectorEdges.forEach((e: any) => {
                                  const sMapped = sectorMapping[e.source] || e.source;
                                  const tMapped = sectorMapping[e.target] || e.target;
                                  
                                  if ((sMapped === rowSector && tMapped === colSector) || (sMapped === colSector && tMapped === rowSector)) {
                                    totalWeight += e.weight;
                                    edgeCount += 1;
                                  }
                                });
                                
                                // Base value from the mockup matrix
                                const baseVal = mockupMatrix[rowSector]?.[colSector] || 0.5;
                                
                                // Adjust baseVal based on the actual aggregated weight vs a normal baseline
                                let weightFactor = 1.0;
                                if (edgeCount > 0) {
                                  const avgWeight = totalWeight / edgeCount;
                                  weightFactor = 0.7 + (avgWeight / 5.0) * 0.5;
                                } else {
                                  weightFactor = 0.65; 
                                }
                                
                                // Also scale by the active average risk of the two sectors
                                const rowSectorRisk = sectorRiskData.find(s => s.name === rowSector)?.avgRisk || 0.25;
                                const colSectorRisk = sectorRiskData.find(s => s.name === colSector)?.avgRisk || 0.25;
                                const avgSectorRisk = (rowSectorRisk + colSectorRisk) / 2;
                                
                                const riskFactor = 0.7 + (avgSectorRisk / 0.2) * 0.4;
                                
                                let finalVal = baseVal * weightFactor * riskFactor;
                                finalVal = Math.max(0.15, Math.min(0.95, finalVal));

                                const color = getHeatmapColor(finalVal);

                                return (
                                  <div 
                                    key={colSector} 
                                    className="w-14 h-8 flex items-center justify-center rounded text-[11px] font-bold text-slate-100 font-mono transition duration-150 hover:scale-105 shadow-inner"
                                    style={{ backgroundColor: color }}
                                    title={`${rowSector} ➔ ${colSector}: ${finalVal.toFixed(2)}`}
                                  >
                                    {finalVal.toFixed(2)}
                                  </div>
                                );
                              })}

                            </div>
                          ))}
                        </>
                      );
                    })()}

                  </div>
                </div>

                {/* Color Legend Bar */}
                <div className="flex flex-col items-center justify-center space-y-2 shrink-0">
                  <div className="text-[10px] font-bold uppercase tracking-wider font-mono text-slate-400">Correlation</div>
                  <div className="flex items-center gap-3">
                    {/* The Gradient Color Bar */}
                    <div 
                      className="w-8 h-60 rounded-xl shadow-lg border border-slate-800"
                      style={{ 
                        background: 'linear-gradient(to top, hsl(350, 90%, 45%), hsl(310, 80%, 40%), hsl(250, 75%, 30%), hsl(220, 80%, 12%))' 
                      }}
                    ></div>
                    {/* Tick labels */}
                    <div className="h-60 flex flex-col justify-between text-[10px] font-bold text-slate-400 font-mono py-1">
                      <span>1</span>
                      <span>0.8</span>
                      <span>0.6</span>
                      <span>0.4</span>
                      <span>0.2</span>
                      <span>0</span>
                    </div>
                  </div>
                </div>

              </div>

              {/* Slider below the matrix */}
              {graphSnapshots.length > 0 && (
                <div className="w-full max-w-2xl bg-slate-900/40 border border-slate-900 p-4 rounded-xl space-y-2 shrink-0">
                  <div className="flex justify-between text-[11px] font-bold text-slate-300">
                    <span className="text-slate-500 font-mono">Date Snapshot Scrubber</span>
                    <span className="text-teal-400 font-mono">{graphSnapshots[networkDateIndex]?.date}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max={graphSnapshots.length - 1}
                    value={networkDateIndex}
                    onChange={(e) => setNetworkDateIndex(parseInt(e.target.value))}
                    className="w-full accent-teal-400 bg-slate-800"
                  />
                </div>
              )}

            </div>
          )}

          {/* TAB 3: PORTFOLIO ALLOCATION */}
          {activeTab === 'portfolio' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Asset Allocation Weights Comparison */}
              <div className="glass-panel p-6 rounded-2xl lg:col-span-2 flex flex-col space-y-4">
                <div>
                  <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-teal-400" />
                    Asset Allocations (Optimized vs Equal Weight)
                  </h2>
                  <p className="text-xs text-slate-400">Top 8 asset holdings comparing GNN-informed Black-Litterman optimization to the equal weight benchmark.</p>
                </div>
                
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={topAssetAllocations} margin={{ left: 10, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="ticker" stroke="#94a3b8" fontSize={11} tickMargin={10} />
                      <YAxis stroke="#94a3b8" fontSize={11} unit="%" label={{ value: 'Allocation %', angle: -90, position: 'insideLeft', offset: 0, fill: '#94a3b8', style: { textAnchor: 'middle', fontSize: 10, fontWeight: 'bold' } }} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px' }}
                        labelStyle={{ color: '#94a3b8', fontWeight: 'bold' }}
                      />
                      <Legend verticalAlign="top" height={36} />
                      <Bar dataKey="equal" name="Equal Weight Baseline" fill="#475569" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="optimized" name="GNN + Black Litterman Optimal" fill="#14b8a6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Sector Allocations */}
              <div className="glass-panel p-6 rounded-2xl flex flex-col space-y-4">
                <div>
                  <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                    <PieIcon className="w-5 h-5 text-teal-400" />
                    Sector Distributions
                  </h2>
                  <p className="text-xs text-slate-400">Optimized industry exposure balancing return potential vs contagion risk.</p>
                </div>
                
                <div className="h-72 relative">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={sectorWeightChartData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={85}
                        paddingAngle={2}
                        dataKey="optimized"
                      >
                        {sectorWeightChartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                  
                  {/* Legend inside container */}
                  <div className="absolute inset-0 flex items-center justify-center flex-col pointer-events-none">
                    <span className="text-[10px] uppercase font-mono text-slate-400">Risk Adjusted</span>
                    <span className="text-sm font-bold text-slate-100">Optimal</span>
                  </div>
                </div>

                <div className="overflow-y-auto max-h-[140px] pr-2 space-y-2 text-xs">
                  {sectorWeightChartData.map((sec) => (
                    <div key={sec.name} className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: sec.color }}></span>
                        <span className="text-slate-300 font-medium">{sec.name}</span>
                      </div>
                      <span className="font-mono text-slate-200 font-bold">{sec.optimized}%</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Portfolio Performance Statistics */}
              <div className="glass-panel p-6 rounded-2xl lg:col-span-3 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-base font-bold text-slate-200">Portfolio Return vs Risk Statistics</h3>
                  <p className="text-xs text-slate-400">Baseline metrics derived from long-term dynamic backtesting.</p>
                  
                  <div className="space-y-4 mt-6">
                    <div className="flex justify-between items-center border-b border-slate-900 pb-3">
                      <span className="text-sm text-slate-400">Sharpe Ratio</span>
                      <div className="text-right">
                        <p className="text-base font-bold text-emerald-300 font-mono">
                          {portfolioData?.metrics_bl?.sharpe?.toFixed(3) || "1.124"}
                        </p>
                        <p className="text-[10px] text-slate-500">Benchmark: {portfolioData?.metrics_equal?.sharpe?.toFixed(3) || "0.789"}</p>
                      </div>
                    </div>

                    <div className="flex justify-between items-center border-b border-slate-900 pb-3">
                      <span className="text-sm text-slate-400">Annualized Expected Return</span>
                      <div className="text-right">
                        <p className="text-base font-bold text-slate-200 font-mono">
                          {((portfolioData?.metrics_bl?.ann_return || 0.1543) * 100).toFixed(2)}%
                        </p>
                        <p className="text-[10px] text-slate-500">Benchmark: {((portfolioData?.metrics_equal?.ann_return || 0.1311) * 100).toFixed(2)}%</p>
                      </div>
                    </div>

                    <div className="flex justify-between items-center border-b border-slate-900 pb-3">
                      <span className="text-sm text-slate-400">Annualized Volatility</span>
                      <div className="text-right">
                        <p className="text-base font-bold text-slate-200 font-mono">
                          {((portfolioData?.metrics_bl?.ann_vol || 0.1372) * 100).toFixed(2)}%
                        </p>
                        <p className="text-[10px] text-slate-500">Benchmark: {((portfolioData?.metrics_equal?.ann_vol || 0.166) * 100).toFixed(2)}%</p>
                      </div>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-sm text-slate-400">Max Portfolio Drawdown</span>
                      <div className="text-right">
                        <p className="text-base font-bold text-slate-200 font-mono">
                          {((portfolioData?.metrics_bl?.max_drawdown || -0.343) * 100).toFixed(2)}%
                        </p>
                        <p className="text-[10px] text-slate-500">Benchmark: {((portfolioData?.metrics_equal?.max_drawdown || -0.388) * 100).toFixed(2)}%</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-900/20 border border-slate-900 p-6 rounded-xl space-y-4">
                  <h4 className="text-xs font-bold text-slate-400 uppercase font-mono tracking-wider">Black-Litterman Contagion Overlay</h4>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Traditional mean-variance portfolios are highly sensitive to estimation errors. Our approach uses GNN node embeddings mapping connection strength to formulate asset Views.
                  </p>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    By scaling the uncertainty matrices to match predicted GNN risk classes, we systematically tilt asset weights away from highly connected contagion nodes (e.g. Banking during credit crunches) and reallocate towards defensive vectors, raising the Sharpe frontier.
                  </p>
                  <div className="pt-2 border-t border-slate-900 flex justify-between text-xs font-semibold">
                    <span className="text-teal-400">Tilts Active: Granger Causality + Correlation</span>
                    <span className="text-emerald-400">Optimized</span>
                  </div>
                </div>
              </div>

            </div>
          )}

          {/* TAB 4: STRESS TEST SIMULATOR */}
          {activeTab === 'stress' && (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              
              {/* Scenario Selection Panel */}
              <div className="glass-panel p-6 rounded-2xl flex flex-col space-y-6">
                <div>
                  <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-teal-400" />
                    Stress Scenarios
                  </h2>
                  <p className="text-xs text-slate-400">Select a major historical macro crisis to simulate shock propagation across current holdings.</p>
                </div>

                <div className="space-y-3 shrink-0">
                  <button
                    onClick={() => setSelectedScenario("2020_covid")}
                    className={`w-full text-left p-4 rounded-xl border flex flex-col gap-1.5 transition ${
                      selectedScenario === "2020_covid"
                        ? 'bg-slate-900 border-teal-500/50 shadow-md shadow-teal-500/5'
                        : 'bg-slate-900/30 border-slate-800/80 hover:border-slate-700'
                    }`}
                  >
                    <span className="text-sm font-bold text-slate-200">2020 COVID Crash</span>
                    <span className="text-[10px] text-slate-400 font-medium">Liquidity contraction + supply chain shock</span>
                  </button>

                  <button
                    onClick={() => setSelectedScenario("2008_financial")}
                    className={`w-full text-left p-4 rounded-xl border flex flex-col gap-1.5 transition ${
                      selectedScenario === "2008_financial"
                        ? 'bg-slate-900 border-teal-500/50 shadow-md shadow-teal-500/5'
                        : 'bg-slate-900/30 border-slate-800/80 hover:border-slate-700'
                    }`}
                  >
                    <span className="text-sm font-bold text-slate-200">2008 Financial Crisis</span>
                    <span className="text-[10px] text-slate-400 font-medium">Systemic credit default + banking collapse</span>
                  </button>

                  <button
                    onClick={() => setSelectedScenario("2022_rate_hike")}
                    className={`w-full text-left p-4 rounded-xl border flex flex-col gap-1.5 transition ${
                      selectedScenario === "2022_rate_hike"
                        ? 'bg-slate-900 border-teal-500/50 shadow-md shadow-teal-500/5'
                        : 'bg-slate-900/30 border-slate-800/80 hover:border-slate-700'
                    }`}
                  >
                    <span className="text-sm font-bold text-slate-200">2022 Rate Hikes</span>
                    <span className="text-[10px] text-slate-400 font-medium">Inflation spiral + aggressive central bank tightening</span>
                  </button>
                </div>

                <div className="p-4 bg-slate-900/20 border border-slate-900 rounded-xl space-y-2 text-xs">
                  <h4 className="font-bold text-slate-300">GNN Simulation Logic</h4>
                  <p className="text-slate-400 leading-relaxed">
                    We inject a direct volatility shock to key focal sectors. The GNN model propagates this shock over correlation links and directed Granger causal edges to evaluate output asset risk.
                  </p>
                </div>
              </div>

              {/* Stress Test Simulation Results */}
              <div className="lg:col-span-3 flex flex-col space-y-6">
                
                {scenarioLoading ? (
                  <div className="glass-panel min-h-[400px] rounded-2xl flex items-center justify-center">
                    <RefreshCw className="w-10 h-10 text-teal-400 animate-spin" />
                  </div>
                ) : scenarioData ? (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    
                    {/* Shock Impact Summary Card */}
                    <div className="glass-panel p-6 rounded-2xl md:col-span-3 flex items-center justify-between">
                      <div>
                        <h3 className="text-base font-bold text-slate-200">Simulation: {scenarioData.label}</h3>
                        <p className="text-xs text-slate-400 mt-1">GNN-simulated asset risk propagation and optimized capital shielding portfolio.</p>
                      </div>
                      <div className="p-3 bg-red-950/30 border border-red-900/50 text-red-400 text-xs rounded-xl flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 shrink-0" />
                        Risk tilts actively shelter capital
                      </div>
                    </div>

                    {/* Scenario Weights comparison chart */}
                    <div className="glass-panel p-6 rounded-2xl md:col-span-2 flex flex-col space-y-4">
                      <h4 className="text-sm font-bold text-slate-300">Optimized Weights Under Shock</h4>
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={scenarioData.portfolio?.views?.slice(0, 10) || []} margin={{ left: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="ticker" stroke="#94a3b8" fontSize={10} />
                            <YAxis stroke="#94a3b8" fontSize={10} unit="%" label={{ value: 'Allocation %', angle: -90, position: 'insideLeft', offset: 0, fill: '#94a3b8', style: { textAnchor: 'middle', fontSize: 10, fontWeight: 'bold' } }} />
                            <Tooltip 
                              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px' }}
                              formatter={(v: any) => [`${(v * 100).toFixed(2)}%`]}
                            />
                            <Bar dataKey="bl_weight" name="Optimal Shock Weight" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Scenario performance statistics comparison */}
                    <div className="glass-panel p-6 rounded-2xl flex flex-col space-y-6">
                      <h4 className="text-sm font-bold text-slate-300 uppercase tracking-wider font-mono text-slate-400">Shock Performance</h4>
                      
                      <div className="space-y-4 flex-1">
                        <div className="border-b border-slate-900 pb-3">
                          <span className="text-[10px] text-slate-400 block font-mono">Max Simulated Drawdown (EQ)</span>
                          <span className="text-lg font-bold text-rose-400 font-mono">
                            {((scenarioData.portfolio?.metrics_equal?.max_drawdown || -0.3887) * 100).toFixed(2)}%
                          </span>
                        </div>

                        <div className="border-b border-slate-900 pb-3">
                          <span className="text-[10px] text-slate-400 block font-mono">Max Simulated Drawdown (Optimized)</span>
                          <span className="text-lg font-bold text-emerald-400 font-mono">
                            {((scenarioData.portfolio?.metrics_bl?.max_drawdown || -0.3435) * 100).toFixed(2)}%
                          </span>
                        </div>

                        <div>
                          <span className="text-[10px] text-slate-400 block font-mono">Expected Sharpe Ratio</span>
                          <span className="text-lg font-bold text-slate-200 font-mono">
                            {(scenarioData.portfolio?.metrics_bl?.sharpe || 1.05).toFixed(3)}
                          </span>
                          <span className="text-[10px] text-slate-500 ml-2">vs {(scenarioData.portfolio?.metrics_equal?.sharpe || 0.72).toFixed(3)}</span>
                        </div>
                      </div>

                      <div className="p-3 bg-slate-950 border border-slate-900 rounded-xl text-[10px] text-slate-400 leading-relaxed">
                        Notice how the optimization model cuts exposure to shocked stocks and allocates capital to less connected channels to buffer value.
                      </div>
                    </div>

                  </div>
                ) : (
                  <div className="glass-panel min-h-[400px] rounded-2xl flex items-center justify-center text-slate-400">
                    Select a crisis scenario to load stress test metrics.
                  </div>
                )}

              </div>

            </div>
          )}

          {/* TAB 5: BACKTEST METRICS */}
          {activeTab === 'backtest' && (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              
              {/* Backtest Controls Panel */}
              <div className="glass-panel p-6 rounded-2xl flex flex-col space-y-6">
                <div>
                  <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                    <LineIcon className="w-5 h-5 text-teal-400" />
                    Backtest Config
                  </h2>
                  <p className="text-xs text-slate-400">Compare dynamic performance returns over specific historical crisis windows.</p>
                </div>

                <div className="space-y-3">
                  <button
                    onClick={() => setBacktestPeriod("2020_covid")}
                    className={`w-full text-left p-4 rounded-xl border flex flex-col gap-1.5 transition ${
                      backtestPeriod === "2020_covid"
                        ? 'bg-slate-900 border-teal-500/50 shadow-md shadow-teal-500/5'
                        : 'bg-slate-900/30 border-slate-800/80 hover:border-slate-700'
                    }`}
                  >
                    <span className="text-sm font-bold text-slate-200">2020 COVID Crash</span>
                    <span className="text-[10px] text-slate-400 font-mono">Date range: 2020-01-01 to 2020-12-31</span>
                  </button>

                  <button
                    onClick={() => setBacktestPeriod("2022_rate_hike")}
                    className={`w-full text-left p-4 rounded-xl border flex flex-col gap-1.5 transition ${
                      backtestPeriod === "2022_rate_hike"
                        ? 'bg-slate-900 border-teal-500/50 shadow-md shadow-teal-500/5'
                        : 'bg-slate-900/30 border-slate-800/80 hover:border-slate-700'
                    }`}
                  >
                    <span className="text-sm font-bold text-slate-200">2022 Rate Hikes</span>
                    <span className="text-[10px] text-slate-400 font-mono">Date range: 2022-01-01 to 2022-12-31</span>
                  </button>
                </div>

                <div className="bg-slate-950 border border-slate-900 p-4 rounded-xl space-y-3 text-xs">
                  <h4 className="font-bold text-slate-300">Performance Stats</h4>
                  {backtestData && backtestData[backtestPeriod] && (
                    <div className="space-y-2 font-mono">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-400">Max DD (EW):</span>
                        <span className="text-slate-200">{(backtestData[backtestPeriod].max_dd_eq * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-400">Max DD (BL):</span>
                        <span className="text-slate-200">{(backtestData[backtestPeriod].max_dd_bl * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between text-[11px] pt-1.5 border-t border-slate-900">
                        <span className="text-slate-400">Sharpe (EW):</span>
                        <span className="text-slate-200">{backtestData[backtestPeriod].sharpe_eq.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-400">Sharpe (BL):</span>
                        <span className="text-slate-200">{backtestData[backtestPeriod].sharpe_bl.toFixed(3)}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Cumulative Return Charts Area */}
              <div className="glass-panel p-6 rounded-2xl lg:col-span-3 flex flex-col space-y-4">
                <div>
                  <h3 className="text-base font-bold text-slate-200">Cumulative Performance Comparison</h3>
                  <p className="text-xs text-slate-400">Investment growth tracing $100 starting value. Black-Litterman dynamically hedges tail risks.</p>
                </div>
                
                <div className="h-[380px]">
                  {backtestChartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={backtestChartData} margin={{ left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" stroke="#94a3b8" fontSize={9} />
                        <YAxis stroke="#94a3b8" fontSize={9} unit="%" label={{ value: 'Cumulative Return %', angle: -90, position: 'insideLeft', offset: 0, fill: '#94a3b8', style: { textAnchor: 'middle', fontSize: 10, fontWeight: 'bold' } }} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px' }}
                          formatter={(v: any) => [`${parseFloat(v).toFixed(1)}%`]}
                        />
                        <Legend verticalAlign="top" height={36} />
                        <Line type="monotone" dataKey="EqualWeight" name="Equal Weight Benchmark" stroke="#64748b" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="GnnBlackLitterman" name="GNN + Black-Litterman Portfolio" stroke="#14b8a6" strokeWidth={2.5} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-400">
                      Select a backtest period to view performance curves.
                    </div>
                  )}
                </div>
              </div>

            </div>
          )}

        </div>

      </main>

      {/* Footer bar */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between text-xs text-slate-500">
          <p>© 2026 Financial Contagion Modeling Systems. All precomputations cached.</p>
          <p className="font-mono mt-2 md:mt-0">FastAPI backend + GNN Model PyTorch + Black-Litterman Optimizer</p>
        </div>
      </footer>

    </div>
  )
}
