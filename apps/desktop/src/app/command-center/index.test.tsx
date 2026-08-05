import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { setSessions } from '@/store/session'

import { CommandCenterView } from './index'

afterEach(() => {
  cleanup()
  setSessions([])
})

describe('CommandCenterView session actions', () => {
  it('passes the clicked row profile when deleting a session', () => {
    const onDeleteSession = vi.fn(async () => undefined)

    const session = {
      id: 'shared-id',
      last_active: 2,
      profile: 'work',
      started_at: 1,
      title: 'Work session'
    } as SessionInfo

    setSessions([session])
    render(
      <MemoryRouter>
        <CommandCenterView
          onClose={vi.fn()}
          onDeleteSession={onDeleteSession}
          onOpenSession={vi.fn()}
        />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Delete session' }))

    expect(onDeleteSession).toHaveBeenCalledWith('shared-id', 'work')
  })
})