import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  getExperiments,
  runExperiment,
  rerunExperiment,
  compareExperiments,
  updateNote,
  runCompare,
} from './client'

function makeResponse(data: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: vi.fn().mockResolvedValue(data),
    text: vi.fn().mockResolvedValue(String(data)),
  } as unknown as Response
}

const TRAINING = { epochs: 5, batch_size: 32, learning_rate: 0.001 }

describe('API client', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
    localStorage.clear()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  // --- auth headers ---

  describe('Authorization header', () => {
    it('is included when token exists in localStorage', async () => {
      localStorage.setItem('token', 'my-jwt')
      vi.mocked(fetch).mockResolvedValueOnce(makeResponse([]))

      await getExperiments()

      const headers = vi.mocked(fetch).mock.calls[0][1]?.headers as Record<string, string>
      expect(headers['Authorization']).toBe('Bearer my-jwt')
    })

    it('is omitted when no token in localStorage', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(makeResponse([]))

      await getExperiments()

      const headers = vi.mocked(fetch).mock.calls[0][1]?.headers as Record<string, string>
      expect(headers['Authorization']).toBeUndefined()
    })
  })

  // --- error handling ---

  describe('handleResponse', () => {
    it('throws with server message on non-ok response', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        text: vi.fn().mockResolvedValue('Unauthorized'),
      } as unknown as Response)

      await expect(getExperiments()).rejects.toThrow('Unauthorized')
    })

    it('throws HTTP status fallback when response body is empty', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: vi.fn().mockResolvedValue(''),
      } as unknown as Response)

      await expect(getExperiments()).rejects.toThrow('HTTP 500')
    })
  })

  // --- getExperiments ---

  describe('getExperiments', () => {
    it('sends GET to /experiments', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(makeResponse([]))

      await getExperiments()

      expect(fetch).toHaveBeenCalledWith('/experiments', expect.any(Object))
      expect(vi.mocked(fetch).mock.calls[0][1]?.method).toBeUndefined()
    })

    it('returns parsed response data', async () => {
      const data = [{ id: 1, model: 'lenet5' }]
      vi.mocked(fetch).mockResolvedValueOnce(makeResponse(data))

      expect(await getExperiments()).toEqual(data)
    })
  })

  // --- runExperiment ---

  describe('runExperiment', () => {
    it('sends POST to /experiments', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(makeResponse({}))

      await runExperiment({ model: 'lenet5', dataset: 'mnist', training: TRAINING })

      expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/experiments')
      expect(vi.mocked(fetch).mock.calls[0][1]?.method).toBe('POST')
    })

    it('serialises the full request body to JSON', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(makeResponse({}))
      const body = { model: 'resnet18', dataset: 'cifar10', training: TRAINING }

      await runExperiment(body)

      expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]?.body as string)).toEqual(body)
    })
  })

  // --- rerunExperiment ---

  describe('rerunExperiment', () => {
    it('sends POST to /experiments/:id/rerun', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(makeResponse({}))

      await rerunExperiment(42)

      expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/experiments/42/rerun')
      expect(vi.mocked(fetch).mock.calls[0][1]?.method).toBe('POST')
    })
  })

  // --- compareExperiments ---

  describe('compareExperiments', () => {
    it('sends POST to /experiments/compare with ids array', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(makeResponse([]))

      await compareExperiments([1, 2, 3])

      expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/experiments/compare')
      expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]?.body as string)).toEqual({ ids: [1, 2, 3] })
    })
  })

  // --- updateNote ---

  describe('updateNote', () => {
    it('sends PATCH to /experiments/:id/note with note in body', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(makeResponse({}))

      await updateNote(7, 'great run')

      expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/experiments/7/note')
      expect(vi.mocked(fetch).mock.calls[0][1]?.method).toBe('PATCH')
      expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]?.body as string)).toEqual({ note: 'great run' })
    })
  })

  // --- runCompare ---

  describe('runCompare', () => {
    it('sends POST to /compare with dataset and training config', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(makeResponse({}))

      await runCompare('mnist', TRAINING)

      expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/compare')
      expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]?.body as string)).toEqual({
        dataset: 'mnist',
        training: TRAINING,
      })
    })
  })
})
