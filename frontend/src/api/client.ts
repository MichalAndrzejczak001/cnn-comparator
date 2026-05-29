import type { ExperimentRequest, ExperimentResponse, CompareResult, TrainingConfig, ClassifyResponse, GradCamResponse } from '../types/api'

function authHeaders(): HeadersInit {
  const token = localStorage.getItem('token')
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const msg = await res.text().catch(() => `HTTP ${res.status}`)
    throw new Error(msg || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export async function getExperiments(): Promise<ExperimentResponse[]> {
  const res = await fetch('/experiments', { headers: authHeaders() })
  return handleResponse(res)
}

export async function runExperiment(body: ExperimentRequest): Promise<ExperimentResponse> {
  const res = await fetch('/experiments', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(body),
  })
  return handleResponse(res)
}

export async function rerunExperiment(id: number): Promise<ExperimentResponse> {
  const res = await fetch(`/experiments/${id}/rerun`, {
    method: 'POST',
    headers: authHeaders(),
  })
  return handleResponse(res)
}

export async function compareExperiments(ids: number[]): Promise<ExperimentResponse[]> {
  const res = await fetch('/experiments/compare', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ ids }),
  })
  return handleResponse(res)
}

export async function updateNote(id: number, note: string): Promise<ExperimentResponse> {
  const res = await fetch(`/experiments/${id}/note`, {
    method: 'PATCH',
    headers: authHeaders(),
    body: JSON.stringify({ note }),
  })
  return handleResponse(res)
}

export async function runCompare(
  dataset: string,
  training: TrainingConfig,
): Promise<CompareResult> {
  const res = await fetch('/compare', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ dataset, training }),
  })
  return handleResponse(res)
}

export async function gradCamImage(id: number, file: File): Promise<GradCamResponse> {
  const token = localStorage.getItem('token')
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`/experiments/${id}/gradcam`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  })
  return handleResponse(res)
}

export async function classifyImage(id: number, file: File): Promise<ClassifyResponse> {
  const token = localStorage.getItem('token')
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`/experiments/${id}/classify`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  })
  return handleResponse(res)
}
