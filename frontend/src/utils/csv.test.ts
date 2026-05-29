import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { csvDate, downloadCsv } from './csv'

describe('csvDate', () => {
  it('returns a string in YYYY-MM-DD format', () => {
    expect(csvDate()).toMatch(/^\d{4}-\d{2}-\d{2}$/)
  })

  it("returns today's date", () => {
    const today = new Date().toISOString().slice(0, 10)
    expect(csvDate()).toBe(today)
  })
})

describe('downloadCsv', () => {
  let capturedContent = ''
  let mockAnchor: { href: string; download: string; click: ReturnType<typeof vi.fn> }

  beforeEach(() => {
    capturedContent = ''
    mockAnchor = { href: '', download: '', click: vi.fn() }

    vi.stubGlobal(
      'Blob',
      class MockBlob {
        constructor(parts: BlobPart[]) {
          capturedContent = parts[0] as string
        }
      },
    )
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:mock-url'),
      revokeObjectURL: vi.fn(),
    })
    vi.spyOn(document, 'createElement').mockReturnValue(
      mockAnchor as unknown as HTMLElement,
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('CSV content starts with BOM character', () => {
    downloadCsv('test.csv', [['a']])
    expect(capturedContent.startsWith('﻿')).toBe(true)
  })

  it('wraps each cell in double quotes', () => {
    downloadCsv('test.csv', [['hello', 'world']])
    expect(capturedContent).toContain('"hello","world"')
  })

  it('escapes double quotes inside cell values', () => {
    downloadCsv('test.csv', [['say "hi"']])
    expect(capturedContent).toContain('"say ""hi"""')
  })

  it('joins rows with CRLF line endings', () => {
    downloadCsv('test.csv', [['row1'], ['row2']])
    expect(capturedContent).toContain('"row1"\r\n"row2"')
  })

  it('converts numbers to strings', () => {
    downloadCsv('test.csv', [[42, 3.14]])
    expect(capturedContent).toContain('"42","3.14"')
  })

  it('sets the download attribute to the provided filename', () => {
    downloadCsv('report.csv', [['x']])
    expect(mockAnchor.download).toBe('report.csv')
  })

  it('calls click() on the anchor element', () => {
    downloadCsv('test.csv', [['x']])
    expect(mockAnchor.click).toHaveBeenCalledOnce()
  })
})
