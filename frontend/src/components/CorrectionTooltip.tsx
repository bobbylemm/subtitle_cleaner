import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'

interface CorrectionTooltipProps {
  children: React.ReactNode
  reason: string
  type: string
}

export function CorrectionTooltip({ children, reason, type }: CorrectionTooltipProps) {
  const [isVisible, setIsVisible] = useState(false)

  const getTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'entity': return 'bg-blue-100 text-blue-800 border-blue-200'
      case 'grammar': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'context': return 'bg-purple-100 text-purple-800 border-purple-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  return (
    <div 
      className="relative inline-block"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      <AnimatePresence>
        {isVisible && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-white dark:bg-zinc-800 rounded-lg shadow-xl border border-zinc-200 dark:border-zinc-700 text-sm"
          >
            <div className={`text-xs font-semibold px-2 py-0.5 rounded-full w-fit mb-1 border ${getTypeColor(type)}`}>
              {type.toUpperCase()}
            </div>
            <p className="text-zinc-600 dark:text-zinc-300 leading-relaxed">
              {reason}
            </p>
            {/* Arrow */}
            <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 border-4 border-transparent border-t-white dark:border-t-zinc-800" />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
