import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import HistoryView from './HistoryView'
import * as client from '../../api/client'
import * as csv from '../../utils/csv'

vi.mock('../../api/client')
vi.mock('../../utils/csv')

const makeExp = (id: number, model = 'lenet5', dataset = 'mnist') => ({
  id,
  model,
  dataset,
  epochs: 5,
  batch_size: 32,
  learning_rate: 0.001,
  train_loss_per_epoch: [0.5, 0.3],
  test_loss_per_epoch: [0.45, 0.28],
  test_loss: 0.28,
  test_accuracy: 0.92,
  training_time_seconds: 12.5,
  confusion_matrix: null,
  note: null,
  created_at: '2024-01-15T10:00:00',
})

const TWO_EXPERIMENTS = [makeExp(1, 'lenet5'), makeExp(2, 'resnet18', 'cifar10')]

describe('HistoryView', () => {
  const onCompareSelected = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(csv.downloadCsv).mockImplementation(() => {})
    vi.mocked(csv.csvDate).mockReturnValue('2024-01-15')
  })

  it('shows loading indicator while fetching', () => {
    vi.mocked(client.getExperiments).mockReturnValue(new Promise(() => {}))
    render(<HistoryView onCompareSelected={onCompareSelected} />)
    expect(screen.getByText('Ładowanie...')).toBeInTheDocument()
  })

  it('renders experiment rows after data loads', async () => {
    vi.mocked(client.getExperiments).mockResolvedValue(TWO_EXPERIMENTS)
    render(<HistoryView onCompareSelected={onCompareSelected} />)
    await waitFor(() => expect(screen.getByText('LeNet-5')).toBeInTheDocument())
    expect(screen.getByText('ResNet-18')).toBeInTheDocument()
  })

  it('shows empty state when no experiments exist', async () => {
    vi.mocked(client.getExperiments).mockResolvedValue([])
    render(<HistoryView onCompareSelected={onCompareSelected} />)
    await waitFor(() =>
      expect(screen.getByText(/Brak eksperymentów/)).toBeInTheDocument(),
    )
  })

  it('displays error message when API call fails', async () => {
    vi.mocked(client.getExperiments).mockRejectedValue(new Error('network error'))
    render(<HistoryView onCompareSelected={onCompareSelected} />)
    await waitFor(() =>
      expect(screen.getByText('Błąd ładowania historii')).toBeInTheDocument(),
    )
  })

  it('hides compare button when fewer than 2 experiments are selected', async () => {
    vi.mocked(client.getExperiments).mockResolvedValue(TWO_EXPERIMENTS)
    render(<HistoryView onCompareSelected={onCompareSelected} />)
    await waitFor(() => screen.getByText('LeNet-5'))
    expect(screen.queryByText(/Porównaj wybrane/)).not.toBeInTheDocument()
  })

  it('shows compare button after selecting 2 experiments', async () => {
    vi.mocked(client.getExperiments).mockResolvedValue(TWO_EXPERIMENTS)
    const user = userEvent.setup()
    render(<HistoryView onCompareSelected={onCompareSelected} />)
    await waitFor(() => screen.getByText('LeNet-5'))

    const checkboxes = screen.getAllByRole('checkbox')
    await user.click(checkboxes[1])
    await user.click(checkboxes[2])

    expect(screen.getByText(/Porównaj wybrane \(2\)/)).toBeInTheDocument()
  })

  it('toggle-all selects every experiment', async () => {
    vi.mocked(client.getExperiments).mockResolvedValue(TWO_EXPERIMENTS)
    const user = userEvent.setup()
    render(<HistoryView onCompareSelected={onCompareSelected} />)
    await waitFor(() => screen.getByText('LeNet-5'))

    await user.click(screen.getByTitle('Zaznacz wszystkie'))

    expect(screen.getByText(/Porównaj wybrane \(2\)/)).toBeInTheDocument()
  })

  it('toggle-all deselects all when all are already selected', async () => {
    vi.mocked(client.getExperiments).mockResolvedValue(TWO_EXPERIMENTS)
    const user = userEvent.setup()
    render(<HistoryView onCompareSelected={onCompareSelected} />)
    await waitFor(() => screen.getByText('LeNet-5'))

    const toggleAll = screen.getByTitle('Zaznacz wszystkie')
    await user.click(toggleAll)
    await user.click(toggleAll)

    expect(screen.queryByText(/Porównaj wybrane/)).not.toBeInTheDocument()
  })
})
