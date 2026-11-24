export const DEFAULT_API = 'https://pasta-rundown-trickle.ngrok-free.dev';
export function apiBase(): string {
  const stored = localStorage.getItem('lab_api');
  // one-time migration: old default localhost -> live Colab URL
  if (!stored || stored.replace(/\/$/, '') === 'http://localhost:8000') {
    localStorage.setItem('lab_api', DEFAULT_API);
    return DEFAULT_API;
  }
  return stored.replace(/\/$/, '');
}
export function setApiBase(v: string) { localStorage.setItem('lab_api', v) }
async function j<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(apiBase() + path, { ...init, headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true', ...(init?.headers || {}) } })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json() as Promise<T>
}
export const Lab = {
  status: () => j<any>('/api/status'),
  profiles: () => j<any>('/api/profiles'),
  formula: (layer: number, neuron: number, dose: number) => j<any>(`/api/formula?layer=${layer}&neuron=${neuron}&dose=${dose}`),
  compare: (b: any) => j<any>('/api/compare', { method: 'POST', body: JSON.stringify(b) }),
  sweep: (b: any) => j<any>('/api/sweep', { method: 'POST', body: JSON.stringify(b) }),
  sweepGet: (q: string) => j<any>('/api/sweep?' + q),
  wordProbs: (b: any) => j<any>('/api/word-probs', { method: 'POST', body: JSON.stringify(b) }),
  randCtrl: (b: any) => j<any>('/api/controls/random', { method: 'POST', body: JSON.stringify(b) }),
  signCtrl: (b: any) => j<any>('/api/controls/sign', { method: 'POST', body: JSON.stringify(b) }),
  gallery: (b: any) => j<any>('/api/gallery', { method: 'POST', body: JSON.stringify(b) }),
}
export function download(name: string, data: any, type = 'application/json') {
  const blob = new Blob([typeof data === 'string' ? data : JSON.stringify(data, null, 2)], { type })
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click()
}
