import { type Metadata } from 'next'

export const metadata: Metadata = {
  title: 'App - Clean Subtitle',
  description: 'Upload your subtitle files (.srt, .vtt) and let our AI correct grammar, entities, and sync issues instantly.',
}

export default function AppLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return <>{children}</>
}
