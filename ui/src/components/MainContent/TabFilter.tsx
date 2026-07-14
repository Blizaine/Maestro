import { useState, useRef, useEffect } from 'react'
import { Heart, Film, Search, X } from 'lucide-react'
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
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (searchOpen && searchRef.current) searchRef.current.focus()
  }, [searchOpen])

  const handleSearchChange = (val: string) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => setSearchQuery(val), 400)
  }

  return (
    <div className="flex items-center gap-1 shrink-0">
      <div className="flex gap-0.5 bg-bg-tertiary rounded-lg p-0.5 border border-border overflow-x-auto shrink-0">
        {tabs.map(tab => (
          <button
            key={tab.value}
            onClick={() => setMediaFilter(tab.value)}
            className={`px-2 md:px-3 py-1 md:py-1.5 rounded-md text-[10px] md:text-xs font-medium transition-all flex items-center gap-1 whitespace-nowrap shrink-0 ${
              mediaFilter === tab.value
                ? tab.value === 'favorites' ? 'bg-red-500/20 text-chip-red'
                : tab.value === 'multiclip' ? 'bg-purple-500/20 text-chip-purple'
                : 'bg-bg-active text-text-primary'
                : 'text-text-muted hover:text-text-secondary'
            }`}
          >
            {tab.icon === 'heart' && <Heart size={11} fill={mediaFilter === 'favorites' ? 'currentColor' : 'none'} />}
            {tab.icon === 'film' && <Film size={11} />}
            <span className="hidden md:inline">{tab.label}</span>
            <span className="md:hidden">{tab.shortLabel}</span>
          </button>
        ))}
      </div>

      {/* Search */}
      {searchOpen ? (
        <div className="flex items-center gap-1 bg-bg-tertiary border border-border rounded-lg px-2 py-0.5">
          <Search size={12} className="text-text-muted shrink-0" />
          <input
            ref={searchRef}
            type="text"
            defaultValue={searchQuery}
            onChange={e => handleSearchChange(e.target.value)}
            placeholder="Search..."
            className="bg-transparent text-xs text-text-primary placeholder:text-text-muted focus:outline-none w-24 md:w-36"
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
        >
          <Search size={14} />
        </button>
      )}
    </div>
  )
}
