import Image from 'next/image'

import { Button } from '@/components/landing/Button'
import { Container } from '@/components/landing/Container'
import backgroundImage from '@/images/landing/background-call-to-action.jpg'

export function CallToAction() {
  return (
    <section
      id="get-started-today"
      className="relative overflow-hidden bg-neutral-900 py-32"
    >
      <Container className="relative">
        <div className="mx-auto max-w-lg text-center">
          <h2 className="font-display text-3xl tracking-tight text-white sm:text-4xl">
            Get started today
          </h2>
          <p className="mt-4 text-lg tracking-tight text-neutral-400">
            It’s time to take control of your subtitles. Try our software so you can
            feel like you’re doing something productive.
          </p>
          <Button href="/app" color="white" className="mt-10">
            Try Demo for Free
          </Button>
        </div>
      </Container>
    </section>
  )
}
