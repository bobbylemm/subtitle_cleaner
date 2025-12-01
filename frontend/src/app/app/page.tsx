'use client'

import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from '@/components/Button'
import clsx from 'clsx'

function CloudArrowUpIcon(props: React.ComponentPropsWithoutRef<'svg'>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" {...props}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z"
      />
    </svg>
  )
}

function CheckCircleIcon(props: React.ComponentPropsWithoutRef<'svg'>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" {...props}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
  )
}

function ArrowDownTrayIcon(props: React.ComponentPropsWithoutRef<'svg'>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" {...props}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"
      />
    </svg>
  )
}

import { diffWords } from 'diff'

import { CorrectionTooltip } from '@/components/CorrectionTooltip'

interface Correction {
  segment_id: number
  original_text: string
  corrected_text: string
  type: string
  reason: string
}

export default function CorrectorPage() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const [originalContent, setOriginalContent] = useState<string>('')

  // Form State
  const [topic, setTopic] = useState('')
  const [industry, setIndustry] = useState('')
  const [country, setCountry] = useState('')

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file: File) => {
    setFile(file)
    setResult(null)
    setError(null)
    
    // Read file content for diff view
    const reader = new FileReader()
    reader.onload = (e) => {
      const text = e.target?.result as string
      setOriginalContent(text)
    }
    reader.readAsText(file)
  }

  const handleSubmit = async () => {
    if (!file) return

    setIsProcessing(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)
    if (topic) formData.append('topic', topic)
    if (industry) formData.append('industry', industry)
    if (country) formData.append('country', country)

    try {
      // Bypass Next.js proxy to avoid timeouts on large files
      const isProd = process.env.NODE_ENV === 'production';
      const defaultUrl = isProd ? 'https://clean-subtitle-tsrxh.ondigitalocean.app' : 'http://localhost:8000';
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || defaultUrl;
      const response = await fetch(`${apiUrl}/v1/universal/universal-correct`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Correction failed. Please try again.')
      }

      const data = await response.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setIsProcessing(false)
    }
  }

  const downloadFile = () => {
    if (!result) return
    const blob = new Blob([result.corrected_content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `corrected_${result.filename}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const resetState = () => {
    setFile(null)
    setResult(null)
    setError(null)
    setOriginalContent('')
    setTopic('')
    setIndustry('')
    setCountry('')
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const getStats = () => {
    if (!result?.applied_corrections) return { entity: 0, grammar: 0, other: 0 }
    
    const stats = {
      entity: 0,
      grammar: 0,
      other: 0
    }

    result.applied_corrections.forEach((c: Correction) => {
      const type = c.type.toLowerCase()
      if (type === 'entity') stats.entity++
      else if (type === 'grammar') stats.grammar++
      else stats.other++
    })

    return stats
  }

  // Diff Logic
  const renderDiff = (original: string, corrected: string, mode: 'original' | 'corrected') => {
    const diff = diffWords(original, corrected)
    const corrections = (result?.applied_corrections || []) as Correction[]

    return diff.map((part, index) => {
      if (mode === 'original') {
        if (part.removed) {
          return (
            <span key={index} className="bg-red-100 text-red-700 decoration-red-500/50 line-through">
              {part.value}
            </span>
          )
        }
        if (!part.added) {
          return <span key={index}>{part.value}</span>
        }
        return null
      } else {
        // Corrected Mode
        if (part.added) {
          // Try to find a matching correction
          const match = corrections.find(c => 
            c.corrected_text.includes(part.value.trim()) && part.value.trim().length > 2
          )

          const content = (
            <span key={index} className="bg-green-100 text-green-700 font-medium cursor-help border-b-2 border-green-200 border-dotted">
              {part.value}
            </span>
          )

          if (match) {
            return (
              <CorrectionTooltip key={index} reason={match.reason} type={match.type}>
                {content}
              </CorrectionTooltip>
            )
          }

          return (
             <span key={index} className="bg-green-100 text-green-700 font-medium">
              {part.value}
            </span>
          )
        }
        if (!part.removed) {
          return <span key={index}>{part.value}</span>
        }
        return null
      }
    })
  }

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16">
      <div className="mx-auto max-w-3xl">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold tracking-tight text-zinc-800 dark:text-zinc-100 sm:text-5xl">
            Clean Subtitle
          </h1>
          <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
            AI-powered correction that preserves slang, style, and context.
          </p>
        </div>

        {/* Main Card / Summary Card */}
        <div className="bg-white dark:bg-zinc-900 rounded-2xl shadow-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden mb-8">
          <div className="p-8">
            {!result ? (
              <>
                {/* Dropzone */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={clsx(
                    'relative flex flex-col items-center justify-center w-full h-64 rounded-xl border-2 border-dashed transition-all cursor-pointer',
                    isDragging
                      ? 'border-emerald-500 bg-emerald-50/50 dark:bg-emerald-900/10'
                      : 'border-zinc-300 dark:border-zinc-700 hover:border-emerald-400 hover:bg-zinc-50 dark:hover:bg-zinc-800/50'
                  )}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileSelect}
                    className="hidden"
                    accept=".srt,.vtt"
                  />
                  
                  {file ? (
                    <div className="text-center">
                      <CheckCircleIcon className="mx-auto h-12 w-12 text-emerald-500" />
                      <p className="mt-4 text-lg font-medium text-zinc-900 dark:text-zinc-100">
                        {file.name}
                      </p>
                      <p className="text-sm text-zinc-500">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                      <p className="mt-2 text-sm text-emerald-600 dark:text-emerald-400">
                        Click to change file
                      </p>
                    </div>
                  ) : (
                    <div className="text-center">
                      <CloudArrowUpIcon className="mx-auto h-12 w-12 text-zinc-400" />
                      <p className="mt-4 text-lg font-medium text-zinc-900 dark:text-zinc-100">
                        Drop your subtitle file here
                      </p>
                      <p className="text-sm text-zinc-500">
                        or click to browse (.srt, .vtt)
                      </p>
                    </div>
                  )}
                </div>

                {/* Advanced Options Toggle */}
                <div className="mt-6">
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center text-sm font-medium text-zinc-600 dark:text-zinc-400 hover:text-emerald-500 transition-colors"
                  >
                    <span className="mr-2">{showAdvanced ? '−' : '+'}</span>
                    Advanced Options
                  </button>

                  <AnimatePresence>
                    {showAdvanced && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                      >
                        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 pt-4">
                          <div>
                            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
                              Topic
                            </label>
                            <input
                              type="text"
                              value={topic}
                              onChange={(e) => setTopic(e.target.value)}
                              placeholder="e.g. Football"
                              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
                              Industry
                            </label>
                            <input
                              type="text"
                              value={industry}
                              onChange={(e) => setIndustry(e.target.value)}
                              placeholder="e.g. Sports"
                              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
                              Country
                            </label>
                            <input
                              type="text"
                              value={country}
                              onChange={(e) => setCountry(e.target.value)}
                              placeholder="e.g. UK"
                              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                            />
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Action Button */}
                <div className="mt-8">
                  <Button
                    onClick={handleSubmit}
                    disabled={!file || isProcessing}
                    className="w-full justify-center py-4 text-lg"
                  >
                    {isProcessing ? 'Processing...' : 'Correct Subtitles'}
                  </Button>
                </div>
              </>
            ) : (
              // Summary Card
              <div className="space-y-8">
                <div className="border-b border-zinc-100 dark:border-zinc-800 pb-6">
                  <div className="mb-4">
                    <h3 className="text-xl font-bold text-zinc-900 dark:text-zinc-100">Processing Complete</h3>
                    <p className="text-sm text-zinc-500 mt-1">
                      {file?.name} • {(file?.size ? file.size / 1024 : 0).toFixed(1)} KB
                    </p>
                  </div>
                  <Button onClick={resetState} variant="outline" className="w-full sm:w-auto text-sm whitespace-nowrap px-8">
                    Process Another File
                  </Button>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="bg-blue-50 dark:bg-blue-900/10 rounded-xl p-4 border border-blue-100 dark:border-blue-800">
                    <div className="text-blue-600 dark:text-blue-400 text-sm font-medium mb-1">Entities Fixed</div>
                    <div className="text-3xl font-bold text-blue-700 dark:text-blue-300">{getStats().entity}</div>
                  </div>
                  <div className="bg-yellow-50 dark:bg-yellow-900/10 rounded-xl p-4 border border-yellow-100 dark:border-yellow-800">
                    <div className="text-yellow-600 dark:text-yellow-400 text-sm font-medium mb-1">Grammar Issues</div>
                    <div className="text-3xl font-bold text-yellow-700 dark:text-yellow-300">{getStats().grammar}</div>
                  </div>
                  <div className="bg-purple-50 dark:bg-purple-900/10 rounded-xl p-4 border border-purple-100 dark:border-purple-800">
                    <div className="text-purple-600 dark:text-purple-400 text-sm font-medium mb-1">Other / Context</div>
                    <div className="text-3xl font-bold text-purple-700 dark:text-purple-300">{getStats().other}</div>
                  </div>
                </div>

                {(topic || industry || country) && (
                  <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-xl p-4 border border-zinc-100 dark:border-zinc-800">
                    <h4 className="text-sm font-medium text-zinc-900 dark:text-zinc-100 mb-3">Settings Applied</h4>
                    <div className="flex flex-wrap gap-2">
                      {topic && (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-zinc-100 text-zinc-800 dark:bg-zinc-700 dark:text-zinc-300">
                          Topic: {topic}
                        </span>
                      )}
                      {industry && (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-zinc-100 text-zinc-800 dark:bg-zinc-700 dark:text-zinc-300">
                          Industry: {industry}
                        </span>
                      )}
                      {country && (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-zinc-100 text-zinc-800 dark:bg-zinc-700 dark:text-zinc-300">
                          Country: {country}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {error && (
              <div className="mt-4 p-4 rounded-md bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm">
                {error}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-8"
        >
          {/* Diff View (Takes up 2 cols) */}
          <div className="lg:col-span-2 bg-white dark:bg-zinc-900 rounded-2xl shadow-xl border border-zinc-200 dark:border-zinc-800 p-8 h-[800px] flex flex-col">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                Correction Complete! ✨
              </h2>
              <div className="flex gap-4">
                <Button onClick={downloadFile} variant="outline">
                  <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
                  Download
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6 flex-1 min-h-0">
              {/* Original Column */}
              <div className="flex flex-col h-full min-h-0 overflow-hidden">
                <h3 className="text-sm font-medium text-zinc-500 uppercase tracking-wider mb-3 flex-none">
                  Original
                </h3>
                <div className="flex-1 min-h-0 overflow-y-auto rounded-xl border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800/50 p-4 font-mono text-sm whitespace-pre-wrap">
                  {renderDiff(originalContent, result.corrected_content, 'original')}
                </div>
              </div>

              {/* Corrected Column */}
              <div className="flex flex-col h-full min-h-0 overflow-hidden">
                <h3 className="text-sm font-medium text-zinc-500 uppercase tracking-wider mb-3 flex-none">
                  Corrected
                </h3>
                <div className="flex-1 min-h-0 overflow-y-auto rounded-xl border border-green-200 dark:border-green-900/50 bg-green-50/30 dark:bg-green-900/10 p-4 font-mono text-sm whitespace-pre-wrap">
                  {renderDiff(originalContent, result.corrected_content, 'corrected')}
                </div>
              </div>
            </div>
          </div>

          {/* Review Panel (Takes up 1 col) */}
          <div className="bg-white dark:bg-zinc-900 rounded-2xl shadow-xl border border-zinc-200 dark:border-zinc-800 p-6 h-[800px] flex flex-col">
             <h3 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 mb-4 flex items-center">
               <span>Corrections Log</span>
               <span className="ml-2 text-xs bg-zinc-100 dark:bg-zinc-800 text-zinc-600 px-2 py-1 rounded-full">
                 {result.applied_corrections?.length || 0}
               </span>
             </h3>
             
             <div className="flex-1 overflow-y-auto pr-2 space-y-3">
               {(result.applied_corrections || []).map((c: Correction, i: number) => (
                 <div key={i} className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-800/30 hover:bg-zinc-100 transition-colors">
                   <div className="flex items-center justify-between mb-1">
                     <span className={clsx(
                       "text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded",
                       c.type === 'entity' ? 'bg-blue-100 text-blue-700' :
                       c.type === 'grammar' ? 'bg-yellow-100 text-yellow-700' :
                       'bg-gray-100 text-gray-700'
                     )}>
                       {c.type}
                     </span>
                     <span className="text-xs text-zinc-400">Seg {c.segment_id}</span>
                   </div>
                   <div className="flex items-center gap-2 text-sm mb-1">
                     <span className="text-red-500 line-through decoration-red-500/30">{c.original_text}</span>
                     <span className="text-zinc-400">→</span>
                     <span className="text-green-600 font-medium">{c.corrected_text}</span>
                   </div>
                   <p className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">
                     {c.reason}
                   </p>
                 </div>
               ))}
               {(!result.applied_corrections || result.applied_corrections.length === 0) && (
                 <div className="text-center text-zinc-400 py-8">
                   No corrections applied.
                 </div>
               )}
             </div>
          </div>
        </motion.div>
      )}
      {/* Pro Tips Section */}
      <div className="mt-16 border-t border-zinc-200 dark:border-zinc-800 pt-16">
        <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100 mb-8 text-center">
          Pro Tips for Perfect Subtitles
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
              Context Matters
            </h3>
            <p className="text-zinc-600 dark:text-zinc-400 text-sm leading-relaxed">
              Providing a <strong>Topic</strong> (e.g., "Medical", "Football") helps our AI understand jargon. Setting the <strong>Industry</strong> refines the tone, ensuring your subtitles sound professional.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
              Regional Accuracy
            </h3>
            <p className="text-zinc-600 dark:text-zinc-400 text-sm leading-relaxed">
              Use the <strong>Country</strong> setting to handle dialect-specific spelling (e.g., "Color" vs "Colour"). This ensures your subtitles feel native to your target audience.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
              Review with Confidence
            </h3>
            <p className="text-zinc-600 dark:text-zinc-400 text-sm leading-relaxed">
              Our <strong>Diff View</strong> highlights every change. Green indicates corrections, while red shows removals. Hover over any green text to see exactly <em>why</em> the change was made.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
