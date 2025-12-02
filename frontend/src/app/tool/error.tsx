'use client'

import { useEffect } from 'react'
import { Button } from '@/components/Button'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error(error)
  }, [error])

  return (
    <div className="flex h-full flex-col items-center justify-center gap-4">
      <h2 className="text-xl font-bold text-zinc-900 dark:text-zinc-100">
        Something went wrong!
      </h2>
      <p className="text-zinc-600 dark:text-zinc-400">
        {error.message || 'An unexpected error occurred.'}
      </p>
      <Button onClick={() => reset()}>Try again</Button>
    </div>
  )
}
