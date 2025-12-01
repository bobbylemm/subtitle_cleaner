import { type Metadata } from 'next'

import { Providers } from '@/app/providers'
import { Layout } from '@/components/Layout'
import { type Section } from '@/components/SectionProvider'

import '@/styles/tailwind.css'
import allSections from '@/generated/sections.json'

export const metadata: Metadata = {
  title: {
    template: '%s - Clean Subtitle',
    default: 'Clean Subtitle - AI-Powered Subtitle Correction',
  },
  description: 'Automatically correct and format your subtitles with AI. Fix grammar, entities, and sync issues while preserving slang and context. Supports .srt and .vtt.',
  keywords: ['subtitle corrector', 'AI subtitle editor', 'fix subtitles', 'SRT cleaner', 'VTT editor', 'subtitle grammar check'],
  authors: [{ name: 'Clean Subtitle Team' }],
  openGraph: {
    title: 'Clean Subtitle - AI-Powered Subtitle Correction',
    description: 'Automatically correct and format your subtitles with AI. Fix grammar, entities, and sync issues.',
    url: 'https://cleansubtitle.com',
    siteName: 'Clean Subtitle',
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Clean Subtitle - AI-Powered Subtitle Correction',
    description: 'Automatically correct and format your subtitles with AI.',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {

  return (
    <html lang="en" className="h-full" suppressHydrationWarning>
      <body className="flex min-h-full bg-white antialiased dark:bg-zinc-900">
        <Providers>
          <div className="w-full">
            <Layout allSections={allSections}>{children}</Layout>
          </div>
        </Providers>
      </body>
    </html>
  )
}
