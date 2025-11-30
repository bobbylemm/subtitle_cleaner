'use client'

import { useEffect, useState } from 'react'
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from '@headlessui/react'
import clsx from 'clsx'

import { Container } from '@/components/landing/Container'

const features = [
  {
    title: 'Entity Correction',
    description:
      "Automatically fix names, places, and technical terms. 'Mecano' becomes 'Upamecano'.",
    icon: (props: any) => (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" {...props}>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
        />
      </svg>
    ),
    render: () => (
      <div className="p-8">
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between border-b border-neutral-100 pb-4">
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-red-400" />
              <span className="font-mono text-sm text-neutral-500 line-through">
                Mecano
              </span>
            </div>
            <div className="text-neutral-300">→</div>
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-green-400" />
              <span className="font-mono text-sm font-medium text-neutral-900">
                Upamecano
              </span>
            </div>
          </div>
          <div className="flex items-center justify-between border-b border-neutral-100 pb-4">
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-red-400" />
              <span className="font-mono text-sm text-neutral-500 line-through">
                Mbape
              </span>
            </div>
            <div className="text-neutral-300">→</div>
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-green-400" />
              <span className="font-mono text-sm font-medium text-neutral-900">
                Mbappé
              </span>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-red-400" />
              <span className="font-mono text-sm text-neutral-500 line-through">
                Man City
              </span>
            </div>
            <div className="text-neutral-300">→</div>
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-green-400" />
              <span className="font-mono text-sm font-medium text-neutral-900">
                Manchester City
              </span>
            </div>
          </div>
        </div>
      </div>
    ),
  },
  {
    title: 'Context Awareness',
    description:
      "Fixes homophones and wrong words based on context. 'The contrast was signed' -> 'The contract was signed'.",
    icon: (props: any) => (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" {...props}>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18"
        />
      </svg>
    ),
    render: () => (
      <div className="p-8">
        <div className="rounded-lg bg-neutral-50 p-4">
          <p className="font-mono text-sm text-neutral-600">
            The <span className="bg-red-100 text-red-700 line-through decoration-red-500/50">contrast</span> was signed yesterday.
          </p>
        </div>
        <div className="mt-4 flex justify-center text-neutral-300">↓</div>
        <div className="mt-4 rounded-lg bg-green-50 p-4 ring-1 ring-green-500/20">
          <p className="font-mono text-sm text-neutral-900">
            The <span className="bg-green-100 text-green-700 font-medium">contract</span> was signed yesterday.
          </p>
        </div>
      </div>
    ),
  },
  {
    title: 'Grammar & Style',
    description:
      "Corrects punctuation, casing, and grammar while preserving your defined style guide.",
    icon: (props: any) => (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" {...props}>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M9.53 16.122a3 3 0 00-5.78 1.128 2.25 2.25 0 01-2.4 2.245 4.5 4.5 0 008.4-2.245c0-.399-.078-.78-.22-1.128zm0 0a15.998 15.998 0 003.388-1.62m-5.043-.025a15.994 15.994 0 011.622-3.395m3.42 3.42a15.995 15.995 0 004.764-4.648l3.876-5.814a1.151 1.151 0 00-1.597-1.597L14.85 6.361a15.996 15.996 0 00-4.647 4.763m0 0c-.39.39-.78.78-1.128 1.172"
        />
      </svg>
    ),
    render: () => (
      <div className="p-8">
        <div className="space-y-4">
          <div className="flex items-start gap-3">
            <div className="mt-1 h-1.5 w-1.5 rounded-full bg-red-400 flex-none" />
            <p className="font-mono text-sm text-neutral-500">
              i think were gonna win the league
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="mt-1 h-1.5 w-1.5 rounded-full bg-green-400 flex-none" />
            <p className="font-mono text-sm text-neutral-900">
              <span className="text-green-600 font-medium">I</span> think <span className="text-green-600 font-medium">we're</span> going to win the league<span className="text-green-600 font-medium">.</span>
            </p>
          </div>
        </div>
      </div>
    ),
  },
  {
    title: 'Hallucination Removal',
    description:
      'Detects and removes ASR hallucinations like "Thank you for watching" or repetitive loops.',
    icon: (props: any) => (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" {...props}>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"
        />
      </svg>
    ),
    render: () => (
      <div className="p-8">
        <div className="space-y-2">
          <p className="font-mono text-sm text-neutral-900">
            So that's how we solved the problem.
          </p>
          <div className="relative">
            <p className="font-mono text-sm text-neutral-300 line-through decoration-red-400 decoration-2">
              Thank you for watching.
            </p>
            <div className="absolute -right-2 top-1/2 -translate-y-1/2 translate-x-full">
              <span className="inline-flex items-center rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800">
                Removed
              </span>
            </div>
          </div>
          <div className="relative">
            <p className="font-mono text-sm text-neutral-300 line-through decoration-red-400 decoration-2">
              Please subscribe.
            </p>
          </div>
        </div>
      </div>
    ),
  },
]

export function PrimaryFeatures() {
  let [tabOrientation, setTabOrientation] = useState<'horizontal' | 'vertical'>(
    'horizontal',
  )

  useEffect(() => {
    let lgMediaQuery = window.matchMedia('(min-width: 1024px)')

    function onMediaQueryChange({ matches }: { matches: boolean }) {
      setTabOrientation(matches ? 'vertical' : 'horizontal')
    }

    onMediaQueryChange(lgMediaQuery)
    lgMediaQuery.addEventListener('change', onMediaQueryChange)

    return () => {
      lgMediaQuery.removeEventListener('change', onMediaQueryChange)
    }
  }, [])

  return (
    <section
      id="features"
      aria-label="Features for running your books"
      className="relative overflow-hidden bg-neutral-900 pt-20 pb-28 sm:py-32"
    >
      <Container className="relative">
        <div className="max-w-2xl md:mx-auto md:text-center xl:max-w-none">
          <h2 className="font-display text-3xl tracking-tight text-white sm:text-4xl md:text-5xl">
            Everything you need for perfect subtitles.
          </h2>
          <p className="mt-6 text-lg tracking-tight text-neutral-400">
            Stop spending hours manually correcting subtitles. Our AI engine handles the tedious work for you.
          </p>
        </div>
        <TabGroup
          className="mt-16 grid grid-cols-1 items-center gap-y-2 pt-10 sm:gap-y-6 md:mt-20 lg:grid-cols-12 lg:pt-0"
          vertical={tabOrientation === 'vertical'}
        >
          {({ selectedIndex }) => (
            <>
              <div className="-mx-4 flex overflow-x-auto pb-4 sm:mx-0 sm:overflow-visible sm:pb-0 lg:col-span-5">
                <TabList className="relative z-10 flex gap-x-4 px-4 whitespace-nowrap sm:mx-auto sm:px-0 lg:mx-0 lg:block lg:gap-x-0 lg:gap-y-1 lg:whitespace-normal">
                  {features.map((feature, featureIndex) => (
                    <div
                      key={feature.title}
                      className={clsx(
                        'group relative rounded-full px-4 py-1 lg:rounded-l-xl lg:rounded-r-none lg:p-6',
                        selectedIndex === featureIndex
                          ? 'bg-white lg:bg-white/10 lg:ring-1 lg:ring-white/10 lg:ring-inset'
                          : 'hover:bg-white/10 lg:hover:bg-white/5',
                      )}
                    >
                      <h3>
                        <Tab
                          className={clsx(
                            'font-display text-lg data-selected:not-data-focus:outline-hidden',
                            selectedIndex === featureIndex
                              ? 'text-green-400 lg:text-white'
                              : 'text-neutral-400 hover:text-white lg:text-white',
                          )}
                        >
                          <span className="absolute inset-0 rounded-full lg:rounded-l-xl lg:rounded-r-none" />
                          {feature.title}
                        </Tab>
                      </h3>
                      <p
                        className={clsx(
                          'mt-2 hidden text-sm lg:block',
                          selectedIndex === featureIndex
                            ? 'text-white'
                            : 'text-neutral-400 group-hover:text-white',
                        )}
                      >
                        {feature.description}
                      </p>
                    </div>
                  ))}
                </TabList>
              </div>
              <TabPanels className="lg:col-span-7">
                {features.map((feature) => (
                  <TabPanel key={feature.title} unmount={false}>
                    <div className="relative sm:px-6 lg:hidden">
                      <div className="absolute -inset-x-4 -top-26 -bottom-17 bg-white/10 ring-1 ring-white/10 ring-inset sm:inset-x-0 sm:rounded-t-xl" />
                      <p className="relative mx-auto max-w-2xl text-base text-white sm:text-center">
                        {feature.description}
                      </p>
                    </div>
                    <div className="mt-10 w-180 overflow-hidden rounded-xl bg-neutral-50 shadow-xl shadow-neutral-900/20 sm:w-auto lg:mt-0 lg:w-271.25">
                      {feature.render()}
                    </div>
                  </TabPanel>
                ))}
              </TabPanels>
            </>
          )}
        </TabGroup>
      </Container>
    </section>
  )
}
