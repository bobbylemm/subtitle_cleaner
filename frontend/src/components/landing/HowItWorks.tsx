import { Container } from '@/components/landing/Container'

export function HowItWorks() {
  return (
    <section
      id="how-it-works"
      aria-label="How it works"
      className="bg-neutral-50 py-20 sm:py-32"
    >
      <Container>
        <div className="mx-auto max-w-2xl md:text-center">
          <h2 className="font-display text-3xl tracking-tight text-neutral-900 sm:text-4xl">
            How Clean Subtitle Works
          </h2>
          <p className="mt-4 text-lg tracking-tight text-neutral-700">
            Our AI-powered platform makes subtitle correction effortless. Here is the simple three-step process to perfect captions.
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-y-12 sm:grid-cols-2 sm:gap-x-6 lg:grid-cols-3 lg:gap-x-8 lg:gap-y-0">
          <div className="text-center md:flex md:items-start md:text-left lg:block lg:text-center">
            <div className="md:shrink-0">
              <div className="flow-root">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-green-100 text-2xl font-bold text-green-600 ring-1 ring-green-600/20 md:mx-0 lg:mx-auto">
                  1
                </div>
              </div>
            </div>
            <div className="mt-6 md:ml-4 md:mt-0 lg:ml-0 lg:mt-6">
              <h3 className="text-base font-semibold leading-7 text-neutral-900">
                Upload Your File
              </h3>
              <p className="mt-2 text-base leading-7 text-neutral-600">
                Simply drag and drop your .srt or .vtt file into our secure uploader. We support all standard subtitle formats used by YouTube, Premiere Pro, and DaVinci Resolve.
              </p>
            </div>
          </div>

          <div className="text-center md:flex md:items-start md:text-left lg:block lg:text-center">
            <div className="md:shrink-0">
              <div className="flow-root">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-green-100 text-2xl font-bold text-green-600 ring-1 ring-green-600/20 md:mx-0 lg:mx-auto">
                  2
                </div>
              </div>
            </div>
            <div className="mt-6 md:ml-4 md:mt-0 lg:ml-0 lg:mt-6">
              <h3 className="text-base font-semibold leading-7 text-neutral-900">
                AI Analysis & Correction
              </h3>
              <p className="mt-2 text-base leading-7 text-neutral-600">
                Our advanced LLM analyzes the context of your video. It fixes grammar, corrects entity spellings (like player names or technical terms), and removes hallucinations.
              </p>
            </div>
          </div>

          <div className="text-center md:flex md:items-start md:text-left lg:block lg:text-center">
            <div className="md:shrink-0">
              <div className="flow-root">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-green-100 text-2xl font-bold text-green-600 ring-1 ring-green-600/20 md:mx-0 lg:mx-auto">
                  3
                </div>
              </div>
            </div>
            <div className="mt-6 md:ml-4 md:mt-0 lg:ml-0 lg:mt-6">
              <h3 className="text-base font-semibold leading-7 text-neutral-900">
                Review & Download
              </h3>
              <p className="mt-2 text-base leading-7 text-neutral-600">
                See exactly what changed with our diff view. Hover over corrections to understand the "why". Once satisfied, export your clean subtitle file instantly.
              </p>
            </div>
          </div>
        </div>
      </Container>
    </section>
  )
}
