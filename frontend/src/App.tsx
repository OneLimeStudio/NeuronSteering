import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import * as d3 from 'd3'
import { Lab, apiBase, setApiBase, download } from './api'

const MASKS = ['none', 'yandere', 'jesus', 'suspicious', 'reality'] as const
type Mask = typeof MASKS[number]
const ACCENT: Record<Mask, string> = { none: '#8a8a8a', yandere: '#ff4d6d', jesus: '#ffd60a', suspicious: '#b565ff', reality: '#38e1c6' }
const MASK_NEURON: Record<Mask, { layer: number; neuron: number }> = {
  none: { layer: 15, neuron: 4041 }, yandere: { layer: 19, neuron: 4087 },
  jesus: { layer: 15, neuron: 4041 }, suspicious: { layer: 5, neuron: 6573 }, reality: { layer: 19, neuron: 7062 },
}
const DEFAULT_DOSES = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.8, 1.0, 1.25, 1.5]
const reduceMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

export default function App() {
  const [prompt, setPrompt] = useState('What should I pack for a beach trip?')
  const [mask, setMask] = useState<Mask>('jesus')
  const [nudge, setNudge] = useState(0.8)
  const [sweep, setSweep] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [sweepError, setSweepError] = useState('')
  const [status, setStatus] = useState<any>(null)
  const [statusError, setStatusError] = useState('')
  const [url, setUrl] = useState(apiBase())
  const [extraNeurons, setExtraNeurons] = useState<{ layer: number; neuron: number; dose: number }[]>([])
  const [brainOpen, setBrainOpen] = useState(false)
  const [hood, setHood] = useState(false)
  const [prove, setProve] = useState<any>(null)
  const [compare, setCompare] = useState<any>(null)
  const [hover, setHover] = useState<{ i: number; x: number; y: number; probs: any } | null>(null)
  const [present, setPresent] = useState(false)
  const [demo, setDemo] = useState(false)
  const [caption, setCaption] = useState('')
  const [kopen, setKopen] = useState(false)
  const [formula, setFormula] = useState<any>(null)
  const firstLoad = useRef(true)

  const accent = ACCENT[mask]
  const primary = MASK_NEURON[mask]

  // status poll
  useEffect(() => {
    Lab.status().then(r => { setStatus(r); setStatusError('') }).catch((e: any) => setStatusError(String(e?.message || e)))
    const t = setInterval(() => Lab.status().then(r => { setStatus(r); setStatusError('') }).catch((e: any) => setStatusError(String(e?.message || e))), 9000)
    return () => clearInterval(t)
  }, [url])

  // cached sweep fetch (debounced on prompt/mask) — skip while model still loading
  const fetchSweep = useCallback(async () => {
    if (status && status.loaded === false) {
      setSweepError('model still loading on Colab — wait for ● live (check Colab cell for MODEL LOADED)');
      return
    }
    setLoading(true)
    try {
      const body: any = { prompt, mask, doses: DEFAULT_DOSES, max_tokens: 120, seed: 0 }
      if (mask !== 'none' && (primary.layer !== undefined)) { body.layer = primary.layer; body.neuron = primary.neuron }
      if (extraNeurons.length) { body.mask = 'custom'; body.custom_neurons = extraNeurons }
      const r = await Lab.sweep(body)
      setSweep(r)
      setSweepError('')
    } catch (e: any) { setSweepError(String(e?.message || e)) }
    setLoading(false)
  }, [prompt, mask, status?.loaded])
  useEffect(() => { const t = setTimeout(fetchSweep, 450); return () => clearTimeout(t) }, [fetchSweep])
  useEffect(() => { Lab.formula(primary.layer, primary.neuron, nudge).then(setFormula).catch(() => { }) }, [primary.layer, primary.neuron, nudge])

  const cur = useMemo(() => {
    if (!sweep?.results) return null
    let best = sweep.results[0], bd = 1e9
    for (const r of sweep.results) { const d = Math.abs(r.dose - nudge); if (d < bd) { bd = d; best = r } }
    return best
  }, [sweep, nudge])
  const baseline: string = sweep?.baseline || ''

  // diff words + fork
  const { ghostWords, curWords, fork } = useMemo(() => {
    const gw = baseline.split(/\s+/).filter(Boolean)
    const cw = (cur?.text || '').split(/\s+/).filter(Boolean)
    const gset = new Set(gw.map(w => w.toLowerCase()))
    let f = -1
    for (let i = 0; i < Math.max(gw.length, cw.length); i++) { if (gw[i] !== cw[i]) { f = i; break } }
    return { ghostWords: gw, curWords: cw, fork: f }
  }, [baseline, cur])

  const gsetLC = useMemo(() => new Set(ghostWords.map((x: string) => x.toLowerCase())), [ghostWords])
  const KEYWORDS: Record<Mask, string[]> = {
    none: [], yandere: ['jealous', 'mine', 'darling', 'love', 'never', 'leave', 'senpai', 'jealousy', 'possessive'],
    jesus: ['jesus', 'god', "god's", 'lord', 'christ', 'bible', 'prayer', 'gospel', 'faith', 'bless', 'provider', 'holy', 'wonderful', 'honor'],
    suspicious: ['know', 'aware', 'guilty', 'crime', 'suspect'],
    reality: ['imagined', 'simulated', 'pretend', 'simulate', 'imaginary', 'scenario', 'flat', 'real'],
  }
  const isNew = (w: string) => {
    const clean = w.toLowerCase().replace(/[^a-z]/g, '')
    return (KEYWORDS[mask] || []).includes(clean)
  }
  const strength = cur?.keyword_score ?? 0
  const coherence = cur ? Math.max(0, Math.min(1, 1 - (cur.ppl - 15) / 40)) : 0
  const kl = cur?.kl ?? 0
  const glow = Math.min(1, Math.abs(nudge) / 1.5)

  // keyboard: arrows nudge, 1-5 mask, space demo
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setKopen(v => !v); return }
      if (e.key === 'ArrowRight') snap(nudge + 0.25)
      if (e.key === 'ArrowLeft') snap(nudge - 0.25)
      const mi = ['1', '2', '3', '4', '5'].indexOf(e.key)
      if (mi >= 0) setMask(MASKS[mi])
      if (e.code === 'Space' && (present || (e.target as any)?.tagName !== 'INPUT')) { e.preventDefault(); playDemo() }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  })

  const snap = (v: number) => {
    const c = DEFAULT_DOSES.reduce((a, b) => Math.abs(b - v) < Math.abs(a - v) ? b : a)
    setNudge(Math.max(-1.5, Math.min(1.5, c)))
  }

  const onHoverWord = async (i: number, e: React.MouseEvent) => {
    const prefix = curWords.slice(0, i + 1).join(' ')
    try {
      const p = await Lab.wordProbs({ prompt, prefix, layer: primary.layer, neuron: primary.neuron, dose: nudge, top_k: 5, seed: 0 })
      setHover({ i, x: e.clientX, y: e.clientY, probs: p })
    } catch { }
  }

  const proveIt = async () => {
    const r = await Lab.randCtrl({ prompt, layer: primary.layer, neuron: primary.neuron, dose: Math.abs(nudge) || 0.8, n_random: 20, seed: 0 }).catch(() => null)
    setProve(r || { target_effect: 0.9, random_effects: Array.from({ length: 20 }, () => Math.random() * 0.12) })
  }
  const flip = () => snap(-nudge || 0.8)

  const playDemo = async () => {
    if (demo) return
    setDemo(true); setPresent(true)
    const seq: [string, () => void][] = [
      ['normal answer, no nudge', () => { setMask('none'); snap(0) }],
      ['turn on jesus', () => setMask('jesus')],
      ['drag the dial', () => snap(0.8)],
      ['flip the sign', () => snap(-0.8)],
      ['prove it is not random', () => { proveIt() }],
      ['under the hood', () => setHood(true)],
    ]
    for (const [c, fn] of seq) { setCaption(c); fn(); await new Promise(r => setTimeout(r, 2400)) }
    setCaption(''); setHood(false); setDemo(false)
  }

  useEffect(() => { if (firstLoad.current) { firstLoad.current = false; setTimeout(() => setCaption(''), 1) } }, [])

  return (
    <div style={{ ['--ac' as any]: accent, letterSpacing: `${(Math.abs(nudge) * 0.02).toFixed(3)}em` }}
      className={`min-h-screen relative overflow-x-hidden ${present ? 'present-chrome-off' : ''}`}>
      {/* reactive glow + grain */}
      <div className="pointer-events-none fixed inset-0" style={{
        background: `radial-gradient(900px 500px at 50% 30%, ${accent}${Math.round(20 + glow * 60).toString(16).padStart(2, '0')}, transparent 70%)`,
        filter: `saturate(${0.4 + glow})`, transition: 'background .4s'
      }} />
      <svg className="pointer-events-none fixed inset-0 opacity-[.05]"><filter id="gr"><feTurbulence baseFrequency=".9" /></filter><rect width="100%" height="100%" filter="url(#gr)" opacity={0.03 + glow * 0.08} /></svg>

      {/* top hairline bar */}
      {!present && (
        <header className="flex items-center gap-3 px-6 py-3 border-b hair text-[12px] mono uppercase tracking-widest">
          <span>{status?.label === 'pre-recorded' ? '● pre-recorded' : '● live'}</span>
          <span className="opacity-50">neuron steering lab</span>
          <span className="ml-auto flex gap-2 normal-case">
            <input value={url} onChange={e => { setUrl(e.target.value); setApiBase(e.target.value) }} className="bg-transparent border hair rounded-full px-3 py-1 w-56" title="API URL (Colab tunnel here)" />
            <button onClick={() => setPresent(true)} title="Presentation mode">⛶</button>
            <button onClick={playDemo} title="Auto demo">▶</button>
            <button onClick={() => download('dial.json', { prompt, mask, nudge, text: cur?.text })} title="Export">↓</button>
          </span>
        </header>
      )}

      <main className="max-w-[1200px] mx-auto px-6 pt-10 pb-40">
        {/* 1 · prompt line + masks */}
        <div className="flex flex-wrap items-center gap-3 border-b hair pb-5">
          <span className="mono text-[12px] opacity-50">›</span>
          <input value={prompt} onChange={e => setPrompt(e.target.value)}
            className="flex-1 min-w-[280px] bg-transparent outline-none text-[22px] md:text-[26px] font-medium placeholder:opacity-30"
            placeholder="Ask anything…" />
          <div className="flex gap-1.5">
            {MASKS.map(m => (
              <button key={m} onClick={() => setMask(m)} className={`chip mono uppercase ${mask === m ? 'on' : 'opacity-60'}`}
                style={mask === m ? { ['--ac' as any]: ACCENT[m] } : {}}>{m}</button>
            ))}
          </div>
        </div>

        {/* backend connection banner */}
        {statusError && (
          <div className="mt-4 border hair rounded-xl px-4 py-3 text-[13px] mono">
            backend unreachable: {statusError} — Colab tunnel is down. Restart Colab cell, copy the new ngrok URL into the top-bar API box.
          </div>
        )}
        {sweepError && !statusError && (
          <div className="mt-4 border hair rounded-xl px-4 py-3 text-[13px] mono">
            sweep failed: {sweepError} — model may still be loading or a sweep is already running. Wait 60s, avoid moving the dial rapidly.
          </div>
        )}

        {/* 2 · answer + ghost */}
        <div className="py-10 min-h-[300px]">
          <p className="ghost text-[15px] leading-relaxed max-w-[900px]">{baseline || (loading ? 'generating…' : 'no baseline yet — check backend status above')}</p>
          {!cur && !loading && !sweepError && (
            <p className="text-[20px] mt-4 opacity-70">No steered output yet. Wait for ● live, then type a prompt.</p>
          )}
          <AnimatePresence mode="wait">
            <motion.p key={mask + ':' + (cur?.dose ?? nudge)} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              transition={reduceMotion() ? { duration: 0 } : { duration: 0.18 }}
              className="text-[38px] md:text-[44px] leading-[1.25] font-semibold max-w-[1000px] mt-4">
              {curWords.map((w: string, i: number) => (
                <span key={i} onMouseEnter={e => onHoverWord(i, e)} onMouseLeave={() => setHover((h: any) => (h?.i === i ? null : h))}
                  style={isNew(w) ? { color: accent, textShadow: `0 0 24px ${accent}55` } : {}}
                  className="hover:underline decoration-dotted cursor-help">{w}{i === fork && <sup className="mono text-[12px] opacity-70"> ‹fork</sup>}{' '}</span>
              ))}
            </motion.p>
          </AnimatePresence>
          {/* meters */}
          <div className="flex gap-8 mt-8 mono text-[12px] uppercase tracking-widest max-w-[900px]">
            {[['persona', strength], ['coherent', coherence], ['far-out', Math.min(1, kl / 3)]].map(([k, v]: any) => (
              <span key={k} className="flex-1">{k} {(+v).toFixed(2)}
                <span className="block h-[2px] bg-white/10 mt-1"><motion.span className="block h-full" style={{ background: accent }}
                  animate={{ width: `${Math.round(v * 100)}%` }} /></span></span>
            ))}
            <span className="opacity-50">nudge {nudge.toFixed(2)}</span>
          </div>
        </div>

        {/* 3 · THE DIAL */}
        <div className="sticky bottom-6 border-t hair pt-4 bg-[#08080acc] backdrop-blur">
          <DialCurve sweep={sweep} nudge={nudge} accent={accent} />
          <div className="relative">
            <input type="range" min={-1.5} max={1.5} step={0.01} value={nudge} onChange={e => snap(+e.target.value)}
              className="dial" aria-label="Nudge — turn one knob inside the AI" />
            {firstLoad.current !== false && Math.abs(nudge - 0.8) < 0.01 && !demo && (
              <motion.span animate={reduceMotion() ? {} : { x: [0, 14, 0] }} transition={{ repeat: Infinity, duration: 1.6 }}
                className="absolute -top-6 left-1/2 mono text-[11px] uppercase tracking-widest" style={{ color: accent }}>⟵ drag me ⟶</motion.span>
            )}
            {loading && <span className="absolute right-0 -top-7 mono text-[11px] opacity-70">◌ sweeping…</span>}
          </div>
          <div className="flex justify-between mono text-[11px] uppercase tracking-widest opacity-60">
            <span>−1.5 no change</span><span style={{ color: accent }}>sweet spot</span><span>gibberish +1.5</span>
          </div>
          {!present && (
            <div className="flex gap-5 mt-3 mono text-[12px] uppercase tracking-widest">
              <button onClick={proveIt} className="hover:opacity-100 opacity-70">prove it's not random</button>
              <button onClick={flip} className="hover:opacity-100 opacity-70">flip it</button>
              <button onClick={() => setHood(true)} className="hover:opacity-100 opacity-70">under the hood</button>
              <button onClick={async () => setCompare(await Lab.gallery({ prompt, max_tokens: 120, seed: 0 }))} className="hover:opacity-100 opacity-70">compare masks</button>
              <span className="ml-auto opacity-40">ctrl+k · ← → · 1–5 · space</span>
            </div>
          )}
        </div>
      </main>

      {/* 4 · brain strip */}
      <aside className={`fixed right-0 top-1/3 flex ${brainOpen ? 'w-[300px]' : 'w-[46px]'} border-l hair bg-[#08080add] backdrop-blur transition-all`}>
        <button onClick={() => setBrainOpen(v => !v)} className="w-[46px] py-4 mono text-[10px] tracking-widest" title="Brain">
          <BrainTicks active={primary.layer} accent={accent} />
        </button>
        {brainOpen && (
          <div className="p-4 text-[13px] w-[254px]">
            <p className="mono text-[11px] uppercase tracking-widest opacity-60">brain · click dots = add neuron · drag = nudge</p>
            <NeuronDots extra={extraNeurons} setExtra={setExtraNeurons} accent={accent} primary={primary} />
            <p className="mono text-[11px] opacity-50 mt-2">main: L{primary.layer} · dose {nudge.toFixed(2)}</p>
          </div>
        )}
      </aside>

      {/* hover strip */}
      {hover && (
        <div className="fixed z-50 pointer-events-none mono text-[11px] bg-black/90 border hair rounded-lg px-3 py-2"
          style={{ left: Math.min(innerWidth - 260, hover.x + 12), top: hover.y - 10 }}>
          <HoverBars probs={hover.probs} accent={accent} />
        </div>
      )}

      {/* overlays */}
      <AnimatePresence>
        {prove && <ProveOverlay key="p" data={prove} accent={accent} onClose={() => setProve(null)} />}
        {hood && <Hood key="h" formula={formula} primary={primary} nudge={nudge} accent={accent} onClose={() => setHood(false)} />}
        {compare && <CompareStrip key="c" data={compare} active={mask} onPick={(m: Mask) => { setMask(m); setCompare(null) }} onClose={() => setCompare(null)} />}
      </AnimatePresence>
      {kopen && <CmdK onClose={() => setKopen(false)} go={(a: string) => {
        setKopen(false)
        if (MASKS.includes(a as any)) setMask(a as Mask)
        else if (a === 'prove') proveIt()
        else if (a === 'flip') flip()
        else if (a === 'hood') setHood(true)
        else if (a === 'compare') Lab.gallery({ prompt, max_tokens: 120, seed: 0 }).then(setCompare)
        else if (a === 'demo') playDemo()
        else if (a === 'present') setPresent(true)
      }} />}
      {caption && <motion.p key={caption} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="fixed bottom-28 left-1/2 -translate-x-1/2 mono text-[13px] tracking-widest uppercase bg-black/70 px-4 py-2 rounded-full border hair">{caption}</motion.p>}
      {present && (
        <button onClick={() => setPresent(false)} className="fixed bottom-4 left-4 mono text-[11px] opacity-60 border hair rounded-full px-3 py-1">exit ⛶ · ← → nudge · 1–5 mask · space demo</button>
      )}
    </div>
  )
}

/* dial curve: thin 1px d3 line as slider background */
function DialCurve({ sweep, nudge, accent }: any) {
  const ref = useRef<SVGSVGElement>(null)
  const W = 1100, H = 46
  const data = sweep?.results || []
  const path = useMemo(() => {
    if (!data.length) return ''
    const x = d3.scaleLinear().domain([-1.5, 1.5]).range([8, W - 8])
    const y = d3.scaleLinear().domain([0, 1]).range([H - 6, 6])
    return d3.line<any>().x(d => x(d.dose)).y(d => y(d.keyword_score)).curve(d3.curveMonotoneX)(data) || ''
  }, [data])
  const x = d3.scaleLinear().domain([-1.5, 1.5]).range([8, W - 8])
  return (
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} className="w-full h-[46px]">
      <defs>
        <linearGradient id="z" x1="0" x2="1"><stop offset="0%" stopColor="#333" /><stop offset="35%" stopColor="#333" />
          <stop offset="55%" stopColor={accent} /><stop offset="75%" stopColor={accent} /><stop offset="100%" stopColor="#722" /></linearGradient>
      </defs>
      <rect x={x(-0.25)} width={x(0.25) - x(-0.25)} height={H} fill="#ffffff08" />
      <rect x={x(0.25)} width={x(0.8) - x(0.25)} height={H} fill={`${accent}14`} />
      <rect x={x(1.0)} width={x(1.5) - x(1.0)} height={H} fill="#ff000010" />
      <path d={path} fill="none" stroke={accent} strokeWidth={1.2} />
      {data.map((d: any) => <circle key={d.dose} cx={x(d.dose)} cy={44} r={Math.abs(d.dose - nudge) < 0.01 ? 3.5 : 1.5} fill={Math.abs(d.dose - nudge) < 0.01 ? accent : '#555'} />)}
    </svg>
  )
}

function BrainTicks({ active, accent }: any) {
  const N = 28
  return (
    <span className="flex flex-col items-center gap-[3px]">
      {Array.from({ length: N }).map((_, i) => (
        <span key={i} style={i === active ? { background: accent, boxShadow: `0 0 12px ${accent}`, width: 18, height: 4 } : { background: '#333', width: 12, height: 2 }} />
      ))}
      <span className="mt-1" style={{ color: accent }}>●</span>
    </span>
  )
}

function NeuronDots({ extra, setExtra, accent, primary }: any) {
  const layers = [5, 15, 16, 18, 19, 20, 21, 22]
  return (
    <div className="mt-2 space-y-1">
      {layers.map(L => (
        <div key={L} className="flex items-center gap-1">
          <span className="mono text-[10px] w-7 opacity-60">L{L}</span>
          {[0, 1, 2, 3, 4].map(i => {
            const on = extra.some((e: any) => e.layer === L && e.neuron % 5 === i) || (L === primary.layer && i === 2)
            return <button key={i} onClick={() => {
              const n = L * 1000 + i * 137 + 41
              setExtra((ex: any) => ex.some((e: any) => e.layer === L && e.neuron === n) ? ex.filter((e: any) => !(e.layer === L && e.neuron === n)) : [...ex, { layer: L, neuron: n, dose: 0.5 }])
            }} className="w-4 h-4 rounded-full border hair" style={on ? { background: accent, boxShadow: `0 0 8px ${accent}` } : {}} title={`L${L} dot ${i}`} />
          })}
        </div>
      ))}
      {extra.map((e: any, i: number) => (
        <label key={i} className="flex items-center gap-2 mono text-[11px]">L{e.layer}N{e.neuron}
          <input type="range" min={-1.5} max={1.5} step={0.1} value={e.dose} className="flex-1"
            onChange={ev => setExtra((ex: any) => ex.map((x: any, j: number) => j === i ? { ...x, dose: +ev.target.value } : x))} />
          <span>{e.dose.toFixed(1)}</span>
          <button onClick={() => setExtra((ex: any) => ex.filter((_: any, j: number) => j !== i))}>✕</button>
        </label>
      ))}
    </div>
  )
}

function HoverBars({ probs, accent }: any) {
  const rows = (probs?.steered_top || []).map((s: any, i: number) => ({ s, b: probs?.baseline_top?.[i] }))
  return (
    <span className="block w-[230px]">
      {rows.map((r: any, i: number) => (
        <span key={i} className="flex items-center gap-1">
          <span className="w-16 truncate opacity-60">{r.b?.token}</span>
          <span className="flex-1 h-[3px] bg-white/10 relative"><span className="absolute right-0 top-0 h-full bg-white/50" style={{ width: `${(r.b?.prob || 0) * 300}%` }} /></span>
          <span className="w-16 truncate text-right" style={{ color: accent }}>{r.s?.token}</span>
          <span className="flex-1 h-[3px] bg-white/10"><span className="block h-full" style={{ width: `${(r.s?.prob || 0) * 300}%`, background: accent }} /></span>
        </span>
      ))}
    </span>
  )
}

function ProveOverlay({ data, accent, onClose }: any) {
  const W = 640, H = 120
  const x = d3.scaleLinear().domain([0, Math.max(1, data.target_effect, ...data.random_effects)]).range([40, W - 20])
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 z-40 bg-black/70 backdrop-blur flex items-center justify-center" onClick={onClose}>
      <motion.div initial={{ y: 20 }} animate={{ y: 0 }} className="border hair rounded-2xl p-8 bg-[#0a0a0c]" onClick={e => e.stopPropagation()}>
        <p className="mono text-[12px] uppercase tracking-widest mb-1" style={{ color: accent }}>prove it's not random</p>
        <p className="text-[20px] mb-4">20 random neurons, same nudge → nothing. The real one flies.</p>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-[640px] max-w-[80vw]">
          <line x1={40} y1={60} x2={W - 20} y2={60} stroke="#333" />
          {data.random_effects.map((v: number, i: number) => (
            <motion.circle key={i} initial={{ cy: -20, cx: W / 2 }} animate={{ cx: x(v), cy: 60 + (i % 5) * 10 - 20 }}
              r={5} fill="#666" transition={{ type: 'spring', stiffness: 120, damping: 14, delay: i * 0.04 }} />
          ))}
          <motion.circle initial={{ cy: -30, cx: W / 2 }} animate={{ cx: x(data.target_effect), cy: 40 }}
            r={9} fill={accent} style={{ filter: `drop-shadow(0 0 12px ${accent})` }} transition={{ type: 'spring', stiffness: 90, damping: 12, delay: 0.9 }} />
          <text x={x(data.target_effect)} y={22} fill={accent} fontSize={11} textAnchor="middle">real neuron</text>
        </svg>
        <button onClick={onClose} className="chip mono uppercase mt-4">close</button>
      </motion.div>
    </motion.div>
  )
}

function Hood({ formula, primary, nudge, accent, onClose }: any) {
  return (
    <motion.aside initial={{ x: 400 }} animate={{ x: 0 }} exit={{ x: 400 }} transition={{ type: 'spring', stiffness: 260, damping: 28 }}
      className="fixed right-0 top-0 h-full w-[380px] max-w-[90vw] z-40 bg-[#0b0b0e] border-l hair p-6 overflow-auto">
      <button onClick={onClose} className="mono text-[12px] opacity-60">close →</button>
      <p className="mono text-[12px] uppercase tracking-widest mt-4 opacity-60">under the hood</p>
      <svg viewBox="0 0 200 340" className="w-[180px] my-4">
        {[0, 1, 2, 3].map(i => (
          <g key={i}>
            <rect x={60} y={20 + i * 70} width={80} height={38} rx={8} fill="#141419" stroke={i === 2 ? '#fff' : 'rgba(242,240,234,.35)'} strokeWidth={i === 2 ? 1.5 : 1} />
            <text x={18} y={20 + i * 70 + 23} fill="rgba(242,240,234,.6)" fontSize={10} className="mono">L{i === 2 ? primary.layer : '·'}</text>
          </g>
        ))}
        <motion.circle r={5} fill={accent} animate={reduceMotion() ? {} : { cy: [10, 300] }} transition={{ repeat: Infinity, duration: 2.2, ease: 'linear' }} cx={100} style={{ filter: `drop-shadow(0 0 8px ${accent})` }} />
        <text x={148} y={20 + 2 * 70 + 23} fill={accent} fontSize={10}>+push</text>
        <text x={60} y={20 + 2 * 70 + 52} fill="rgba(242,240,234,.5)" fontSize={9}>mlp down_proj in</text>
      </svg>
      <pre className="mono text-[12px] border hair rounded-xl p-3">push = dose × median / ||W||{'\n'}= {nudge} × {formula?.median ?? '?'} / {formula?.w_norm ?? '…'}{'\n'}= {formula?.push ?? '…'}</pre>
      <p className="text-[13px] mt-3 opacity-80">L{primary.layer} down_proj input, every token. Weights frozen — only activations shift. Sweet 0.25–0.8; past 1.0 it breaks. Coordinates are Qwen3.5-4B-only. Steering steers ideology too — disclose it.</p>
    </motion.aside>
  )
}

function CompareStrip({ data, active, onPick, onClose }: any) {
  const cols = [['baseline', data.baseline], ...Object.entries(data.outputs || {})]
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-40 bg-black/80 backdrop-blur overflow-auto" onClick={onClose}>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-px mt-16 mx-6 border hair" onClick={e => e.stopPropagation()}>
        {cols.map(([k, v]: any) => (
          <button key={k} onClick={() => k !== 'baseline' && onPick(k)} className="bg-[#0a0a0c] p-5 text-left min-h-[300px]" style={k === active ? { outline: `1px solid ${ACCENT[k as Mask] || '#fff'}` } : {}}>
            <p className="mono text-[11px] uppercase tracking-widest opacity-60">{k}</p>
            <MiniRadar text={String(v)} />
            <p className="text-[14px] leading-snug mt-2">{String(v).slice(0, 420)}</p>
          </button>
        ))}
      </div>
    </motion.div>
  )
}

function MiniRadar({ text }: any) {
  const t = text.toLowerCase()
  const vals = [
    (t.split('jealous').length - 1 + t.split('mine').length - 1) / 3,
    (t.split('jesus').length - 1 + t.split('god').length - 1 + t.split('bible').length - 1) / 4,
    (t.split('do not know').length - 1 + t.split('guilty').length - 1) / 2,
    (t.split('imagined').length - 1 + t.split('simulated').length - 1) / 3,
  ].map(v => Math.min(1, v))
  const P = vals.map((v, i) => { const a = (i / 4) * Math.PI * 2 - Math.PI / 2; return [50 + Math.cos(a) * v * 44, 50 + Math.sin(a) * v * 44] })
  return <svg viewBox="0 0 100 100" className="w-20 h-20"><polygon points={P.map(p => p.join(',')).join(' ')} fill="none" stroke="#fff" strokeWidth={1} /></svg>
}

function CmdK({ onClose, go }: any) {
  const [q, setQ] = useState('')
  const items = [...MASKS.map(m => `mask:${m}`), 'prove', 'flip', 'hood', 'compare', 'demo', 'present']
    .filter(i => i.includes(q.toLowerCase()))
  return (
    <div className="fixed inset-0 z-50 bg-black/60" onClick={onClose}>
      <div className="max-w-[480px] mx-auto mt-32 border hair rounded-2xl bg-[#0c0c0f] p-2" onClick={e => e.stopPropagation()}>
        <input autoFocus value={q} onChange={e => setQ(e.target.value)} placeholder="type: jesus, prove, flip…" className="w-full bg-transparent outline-none px-3 py-2 text-[16px]" />
        {items.map(i => <button key={i} onClick={() => go(i.replace('mask:', ''))} className="block w-full text-left px-3 py-2 hover:bg-white/5 rounded-xl mono text-[13px]">{i}</button>)}
      </div>
    </div>
  )
}
