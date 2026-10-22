import { Suspense, useState, type ReactNode } from 'react'

/** Preserve a panel's state after first use, without booting hidden panels. */
export function DeferredPanel({active, children}: {active: boolean; children: ReactNode}) {
  const [opened, setOpened] = useState(active)
  if (active && !opened) setOpened(true)
  if (!active && !opened) return null
  return <Suspense fallback={null}>{children}</Suspense>
}
