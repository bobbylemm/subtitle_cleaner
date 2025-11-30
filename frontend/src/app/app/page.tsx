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
      const response = await fetch('/api/v1/universal/universal-correct', {
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

  // Diff Logic
  const renderDiff = (original: string, corrected: string, mode: 'original' | 'corrected') => {
    const diff = diffWords(original, corrected)

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

        {/* Main Card */}
        <div className="bg-white dark:bg-zinc-900 rounded-2xl shadow-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden mb-8">
          <div className="p-8">
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
          className="mt-8"
        >
          <div className="bg-white dark:bg-zinc-900 rounded-2xl shadow-xl border border-zinc-200 dark:border-zinc-800 p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                Correction Complete! ✨
              </h2>
              <div className="flex gap-4">
                <div className="text-sm text-zinc-500 flex items-center">
                  <span className="inline-block w-3 h-3 bg-green-100 border border-green-300 rounded mr-2"></span>
                  <span>Added</span>
                  <span className="inline-block w-3 h-3 bg-red-100 border border-red-300 rounded ml-4 mr-2"></span>
                  <span>Removed</span>
                </div>
                <Button onClick={downloadFile} variant="outline">
                  <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
                  Download Corrected File
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6 h-[600px]">
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
        </motion.div>
      )}
    </div>
  )
}
