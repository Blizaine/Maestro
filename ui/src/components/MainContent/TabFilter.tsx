import { useState, useRef, useEffect } from 'react'
import { ChevronLeft, ChevronRight, Heart, Film, Search, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import type { MediaFilter } from '../../types'

const tabs: { value: MediaFilter; label: string; shortLabel: string; icon?: string }[] = [
  { value: 'all', label: 'All', shortLabel: 'All' },
  { value: 'images', label: 'Images', shortLabel: 'Img' },
  { value: 'videos', label: 'Videos', shortLabel: 'Vid' },
  { value: 'audio', label: 'Audio', shortLabel: 'Aud' },
  { value: 'avatars', label: 'Edits', shortLabel: 'Edit' },
  { value: 'multiclip', label: 'Multi-clip', shortLabel: 'MC', icon: 'film' },
  { value: 'favorites', label: 'Favorites', shortLabel: '', icon: 'heart' },
]

export function TabFilter() {
  const mediaFilter = useStore(s => s.mediaFilter)
  const setMediaFilter = useStore(s => s.setMediaFilter)
  const searchQuery = useStore(s => s.outputSearchQuery)
  const setSearchQuery = useStore(s => s.setOutputSearchQuery)
  const [searchOpen, setSearchOpen] = useState(false)
  const searchRef = useRef<HTMLInputElement>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const tabsScrollRef = useRef<HTMLDivElement>(null)
  const selectedTabRef = useRef<HTMLButtonElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [canScrollLeft, setCanScrollLeft] = useState(false)
  const [canScrollRight, setCanScrollRight] = useState(false)

  useEffect(() => {
    if (searchOpen && searchRef.current) searchRef.current.focus()
  }, [searchOpen])

  useEffect(() => {
    selectedTabRef.current?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }, [mediaFilter])

  useEffect(() => {
    const element = tabsScrollRef.current
    if (!element) return

    const updateScrollEdges = () => {
      setCanScrollLeft(element.scrollLeft > 2)
      setCanScrollRight(element.scrollLeft + element.clientWidth < element.scrollWidth - 2)
    }

    updateScrollEdges()
    element.addEventListener('scroll', updateScrollEdges, { passive: true })
    window.addEventListener('resize', updateScrollEdges)
    const observer = typeof ResizeObserver !== 'undefined'
      ? new ResizeObserver(updateScrollEdges)
      : null
    observer?.observe(element)

    return () => {
      element.removeEventListener('scroll', updateScrollEdges)
      window.removeEventListener('resize', updateScrollEdges)
      observer?.disconnect()
    }
  }, [])

  useEffect(() => {
    if (!searchOpen) return
    const handleOutsideClick = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setSearchOpen(false)
      }
    }
    document.addEventListener('mousedown', handleOutsideClick)
    return () => document.removeEventListener('mousedown', handleOutsideClick)
  }, [searchOpen])

  useEffect(() => () => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
  }, [])

  const handleSearchChange = (val: string) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => setSearchQuery(val), 400)
  }

  return (
    <div ref={rootRef} className="relative flex min-w-0 flex-1 items-center gap-1">
      <div className="relative w-fit min-w-0 max-w-full shrink">
        <div ref={tabsScrollRef} className="flex w-max max-w-full gap-0.5 overflow-x-auto rounded-lg border border-border bg-bg-tertiary p-0.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {tabs.map(tab => (
            <button
              key={tab.value}
              ref={mediaFilter === tab.value ? selectedTabRef : undefined}
              onClick={() => setMediaFilter(tab.value)}
              className={`flex shrink-0 items-center gap-1 whitespace-nowrap rounded-md px-2 py-1 text-[10px] font-medium transition-all xl:px-3 xl:py-1.5 xl:text-xs ${
                mediaFilter === tab.value
                  ? tab.value === 'favorites' ? 'bg-red-500/20 text-chip-red'
                  : tab.value === 'multiclip' ? 'bg-purple-500/20 text-chip-purple'
                  : 'bg-bg-active text-text-primary'
                  : 'text-text-muted hover:text-text-secondary'
              }`}
              title={tab.label}
            >
              {tab.icon === 'heart' && <Heart size={11} fill={mediaFilter === 'favorites' ? 'currentColor' : 'none'} />}
              {tab.icon === 'film' && <Film size={11} />}
              <span className="hidden xl:inline">{tab.label}</span>
              <span className="xl:hidden">{tab.shortLabel}</span>
            </button>
          ))}
        </div>
        {canScrollLeft && (
          <>
            <div className="pointer-events-none absolute inset-y-px left-px hidden w-7 rounded-l-lg bg-linear-to-r from-bg-tertiary to-transparent md:block xl:hidden" />
            <button
              type="button"
              onClick={() => tabsScrollRef.current?.scrollBy({ left: -140, behavior: 'smooth' })}
              className="absolute left-1 top-1/2 hidden -translate-y-1/2 items-center justify-center rounded-full bg-bg-secondary/95 p-0.5 text-text-secondary shadow-md hover:text-text-primary md:flex xl:hidden"
              aria-label="Scroll gallery filters left"
            >
              <ChevronLeft size={12} />
            </button>
          </>
        )}
        {canScrollRight && (
          <>
            <div className="pointer-events-none absolute inset-y-px right-px hidden w-7 rounded-r-lg bg-linear-to-l from-bg-tertiary to-transparent md:block xl:hidden" />
            <button
              type="button"
              onClick={() => tabsScrollRef.current?.scrollBy({ left: 140, behavior: 'smooth' })}
              className="absolute right-1 top-1/2 hidden -translate-y-1/2 items-center justify-center rounded-full bg-bg-secondary/95 p-0.5 text-text-secondary shadow-md hover:text-text-primary md:flex xl:hidden"
              aria-label="Scroll gallery filters right"
            >
              <ChevronRight size={12} />
            </button>
          </>
        )}
      </div>

      {/* Search */}
      {searchOpen ? (
        <div className="flex items-center gap-1 rounded-lg border border-border bg-bg-tertiary px-2 py-0.5 md:absolute md:right-0 md:top-full md:z-50 md:mt-2 md:w-56 md:shadow-xl xl:static xl:mt-0 xl:w-auto xl:shadow-none">
          <Search size={12} className="text-text-muted shrink-0" />
          <input
            ref={searchRef}
            type="text"
            defaultValue={searchQuery}
            onChange={e => handleSearchChange(e.target.value)}
            onKeyDown={event => { if (event.key === 'Escape') setSearchOpen(false) }}
            placeholder="Search..."
            className="w-24 bg-transparent text-xs text-text-primary placeholder:text-text-muted focus:outline-none md:min-w-0 md:flex-1 xl:w-36 xl:flex-none"
          />
          <button onClick={() => { setSearchOpen(false); if (searchQuery) setSearchQuery('') }}
            className="text-text-muted hover:text-text-secondary">
            <X size={12} />
          </button>
        </div>
      ) : (
        <button
          onClick={() => setSearchOpen(true)}
          className={`p-1.5 rounded-lg transition-colors ${searchQuery ? 'text-accent-blue bg-accent-blue/10' : 'text-text-muted hover:text-text-secondary hover:bg-bg-hover'}`}
          title="Search outputs"
          aria-label="Search outputs"
        >
          <Search size={14} />
        </button>
      )}
    </div>
  )
}
