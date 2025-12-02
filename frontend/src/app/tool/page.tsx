import { type Metadata } from 'next'
import CorrectorClient from './CorrectorClient'

export const dynamic = 'force-dynamic'

export const metadata: Metadata = {
  title: 'App - Clean Subtitle',
  description: 'AI-powered subtitle correction tool. Fix grammar, sync, and formatting issues in SRT and VTT files.',
}

export default function AppPage() {
  return <CorrectorClient />
}
