import { useEffect, useState } from 'react'
import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react'
import {
  subscribeMaestroAlerts,
  type MaestroAlert,
} from '../lib/notifications'

const TOAST_LIFETIME_MS = 6500

export function NotificationToastHost() {
  const [alerts, setAlerts] = useState<MaestroAlert[]>([])

  useEffect(() => subscribeMaestroAlerts(alert => {
    setAlerts(current => [...current.slice(-3), alert])
    window.setTimeout(() => {
      setAlerts(current => current.filter(item => item.id !== alert.id))
    }, TOAST_LIFETIME_MS)
  }), [])

  if (alerts.length === 0) return null

  return (
    <div
      className="fixed right-3 top-14 z-[95] flex w-[min(360px,calc(100vw-1.5rem))] flex-col gap-2 md:right-5 md:top-5"
      aria-live="polite"
      aria-atomic="false"
    >
      {alerts.map(alert => {
        const failed = alert.category === 'failure'
        const cancelled = alert.category === 'cancelled'
        const Icon = failed ? AlertTriangle : cancelled ? Info : CheckCircle2
        return (
          <div
            key={alert.id}
            role={failed ? 'alert' : 'status'}
            className={`flex items-start gap-2.5 rounded-xl border bg-bg-secondary/95 px-3 py-2.5 shadow-2xl backdrop-blur ${
              failed ? 'border-red-500/40' : 'border-border'
            }`}
          >
            <Icon
              size={16}
              className={`mt-0.5 shrink-0 ${
                failed
                  ? 'text-red-400'
                  : cancelled
                    ? 'text-text-muted'
                    : 'text-indicator-success'
              }`}
            />
            <div className="min-w-0 flex-1">
              <div className="text-xs font-medium text-text-primary">{alert.title}</div>
              <div className="mt-0.5 text-[10px] leading-relaxed text-text-secondary">{alert.body}</div>
            </div>
            <button
              type="button"
              onClick={() => setAlerts(current => current.filter(item => item.id !== alert.id))}
              className="rounded p-0.5 text-text-muted hover:bg-bg-hover hover:text-text-primary"
              aria-label="Dismiss notification"
            >
              <X size={12} />
            </button>
          </div>
        )
      })}
    </div>
  )
}
