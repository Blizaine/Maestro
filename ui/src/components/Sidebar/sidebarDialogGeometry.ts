export type SidebarDialogVariant = 'settings' | 'library' | 'workflow'

type Rect = {
  left: number
  right: number
  top: number
  bottom: number
  width: number
}

export type SidebarDialogPosition = {
  left?: number
  top?: number
  bottom?: number
  width?: number
  height?: number
  maxHeight?: number
}

export function getSidebarDialogPosition({
  variant,
  viewportTop,
  viewportHeight,
  windowWidth,
  windowHeight,
  mobile = windowWidth < 768,
  sidebar,
  settings,
  trigger,
  fixedHeight,
}: {
  variant: SidebarDialogVariant
  viewportTop: number
  viewportHeight: number
  windowWidth: number
  windowHeight: number
  mobile?: boolean
  sidebar?: Rect | null
  settings?: Pick<Rect, 'top'> | null
  trigger?: Rect | null
  fixedHeight?: number
}): SidebarDialogPosition {
  let bounds: SidebarDialogPosition

  if (mobile && variant === 'settings') {
    // Open upward near the footer controls. A keyboard or iOS viewport pan can
    // leave that anchor offscreen or too little room for text entry; in that
    // case use the visible viewport's bottom, retaining the dialog's scroller.
    const left = Math.max(8, (sidebar?.left ?? 0) + 8)
    const right = Math.min(windowWidth - 8, (sidebar?.right ?? windowWidth) - 8)
    const visibleTop = viewportTop + 8
    const visibleBottom = viewportTop + viewportHeight - 8
    const anchorTop = trigger?.top ?? settings?.top
    const candidateBottom = anchorTop == null ? visibleBottom : Math.min(anchorTop - 6, visibleBottom)
    const minimumSpace = Math.min(240, Math.max(0, viewportHeight - 16))
    const bottomEdge = candidateBottom - visibleTop >= minimumSpace ? candidateBottom : visibleBottom
    bounds = {
      left,
      bottom: windowHeight - bottomEdge,
      width: Math.max(0, right - left),
      maxHeight: Math.max(0, bottomEdge - visibleTop),
    }
  } else if (variant === 'settings' && trigger && sidebar) {
    const width = Math.min(360, sidebar.width - 24, windowWidth - 24)
    const bottomEdge = Math.min(trigger.top, viewportTop + viewportHeight - 8)
    bounds = {
      left: Math.max(sidebar.left + 12, Math.min(trigger.right - width, sidebar.right - width - 12)),
      bottom: windowHeight - bottomEdge + 6,
      width,
      maxHeight: Math.max(44, bottomEdge - viewportTop - 18),
    }
  } else if (mobile) {
    const left = variant === 'settings' && sidebar ? Math.max(8, sidebar.left + 8) : 8
    const right = variant === 'settings' && sidebar ? Math.min(windowWidth - 8, sidebar.right - 8) : windowWidth - 8
    bounds = {
      left,
      width: right - left,
      bottom: Math.max(0, windowHeight - viewportHeight - viewportTop) + 8,
      maxHeight: viewportHeight - 16,
    }
  } else if (variant === 'workflow' && trigger) {
    bounds = {
      left: trigger.left,
      top: trigger.bottom + 6,
      width: trigger.width,
      maxHeight: Math.max(120, viewportHeight + viewportTop - trigger.bottom - 18),
    }
  } else if (variant === 'library' && sidebar && windowWidth - sidebar.right >= 324) {
    bounds = {
      left: sidebar.right + 12,
      top: viewportTop + 12,
      width: Math.min(520, windowWidth - sidebar.right - 24),
      maxHeight: viewportHeight - 24,
    }
  } else {
    const bottomEdge = Math.max(viewportTop + 180, Math.min(settings?.top ?? viewportHeight - 80, viewportTop + viewportHeight - 8))
    bounds = {
      left: (sidebar?.left || 0) + 12,
      bottom: windowHeight - bottomEdge + 8,
      width: Math.min((sidebar?.width || 420) - 24, windowWidth - 24),
      maxHeight: bottomEdge - viewportTop - 20,
    }
  }

  if (fixedHeight != null) bounds.height = Math.min(fixedHeight, Number(bounds.maxHeight))
  return bounds
}
