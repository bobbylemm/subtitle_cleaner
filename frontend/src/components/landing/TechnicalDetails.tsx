import { Container } from '@/components/landing/Container'
import { FadeIn } from '@/components/landing/FadeIn'

export function TechnicalDetails() {
  return (
    <section className="mt-24 sm:mt-32 lg:mt-40">
      <Container>
        <FadeIn>
          <div className="max-w-2xl">
            <h2 className="font-display text-3xl font-medium tracking-tight text-neutral-900 sm:text-4xl">
              Under the hood: Context-Aware AI.
            </h2>
            <p className="mt-6 text-lg tracking-tight text-neutral-600">
              Most spell checkers fail because they look at words in isolation. Clean Subtitle understands the entire conversation.
            </p>
          </div>
        </FadeIn>

        <div className="mt-16 space-y-20 lg:mt-24 lg:space-y-32">
          <FadeIn>
            <article>
              <div className="grid grid-cols-1 gap-8 lg:grid-cols-2 lg:gap-16">
                <div className="relative h-64 overflow-hidden rounded-3xl bg-neutral-100 lg:h-auto">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-sm font-mono text-neutral-500 mb-2">Standard Spell Check</div>
                      <div className="text-xl font-bold text-red-500 line-through">The meet is at 5.</div>
                      <div className="mt-4 text-sm font-mono text-neutral-500 mb-2">Clean Subtitle AI</div>
                      <div className="text-xl font-bold text-green-600">The <span className="underline decoration-green-400">meat</span> is at 5.</div>
                      <div className="mt-2 text-xs text-neutral-400">(Context: A cooking show)</div>
                    </div>
                  </div>
                </div>
                <div>
                  <h3 className="font-display text-2xl font-semibold text-neutral-900">
                    Semantic Understanding
                  </h3>
                  <p className="mt-6 text-lg text-neutral-600">
                    Our engine doesn't just check if a word exists in the dictionary. It analyzes the semantic meaning of the sentence. If you are talking about cooking, it knows "steak" is more likely than "stake". If you are discussing business, it prefers "stake". This disambiguation is powered by a massive context window that considers previous and future sentences.
                  </p>
                  <p className="mt-4 text-lg text-neutral-600">
                    This is crucial for subtitles, where homophones (words that sound the same but have different meanings) are the most common source of errors in automated transcription.
                  </p>
                </div>
              </div>
            </article>
          </FadeIn>

          <FadeIn>
            <article>
              <div className="grid grid-cols-1 gap-8 lg:grid-cols-2 lg:gap-16">
                <div className="lg:order-last relative h-64 overflow-hidden rounded-3xl bg-neutral-100 lg:h-auto">
                   <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-sm font-mono text-neutral-500 mb-2">Input</div>
                      <div className="text-xl font-bold text-neutral-800">...gonna go to the store...</div>
                      <div className="mt-4 text-sm font-mono text-neutral-500 mb-2">Strict Mode</div>
                      <div className="text-xl font-bold text-neutral-800">...going to go to the store...</div>
                      <div className="mt-4 text-sm font-mono text-neutral-500 mb-2">Preservation Mode</div>
                      <div className="text-xl font-bold text-green-600">...gonna go to the store...</div>
                    </div>
                  </div>
                </div>
                <div>
                  <h3 className="font-display text-2xl font-semibold text-neutral-900">
                    Dialect & Slang Preservation
                  </h3>
                  <p className="mt-6 text-lg text-neutral-600">
                    A common frustration with tools like Grammarly is that they try to "fix" your style. If a character in a movie says "Y'all ain't ready," you don't want it corrected to "You all are not ready."
                  </p>
                  <p className="mt-4 text-lg text-neutral-600">
                    Clean Subtitle respects the speaker's voice. We have trained our models to distinguish between an actual error (like "teh" instead of "the") and a stylistic choice (like "gonna" or "wanna"). You can even tune this strictness in our advanced settings.
                  </p>
                </div>
              </div>
            </article>
          </FadeIn>
        </div>
      </Container>
    </section>
  )
}
