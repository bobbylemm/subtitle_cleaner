import { Container } from '@/components/landing/Container'
import { FadeIn } from '@/components/landing/FadeIn'

const useCases = [
  {
    title: 'Content Creators & YouTubers',
    description:
      'Auto-generated captions from YouTube or Premiere Pro are often riddled with errors. Clean Subtitle fixes them instantly, ensuring your "Skibidi" doesn\'t become "Scooby Doo". Boost your SEO and audience retention with accurate subtitles.',
  },
  {
    title: 'Filmmakers & Editors',
    description:
      'Post-production is stressful enough without worrying about typos in your SRT files. Our tool integrates seamlessly into your workflow, handling industry-standard formats and preserving your precise timecodes for DaVinci Resolve or Final Cut Pro.',
  },
  {
    title: 'Educational Institutions',
    description:
      'Accessibility is a legal requirement for many universities. Ensure your lecture videos have 100% accurate captions for hearing-impaired students. Our AI handles complex academic terminology and technical jargon with ease.',
  },
  {
    title: 'Podcasters',
    description:
      'Video podcasts are exploding on Spotify and YouTube. Don\'t let bad captions ruin the experience. We handle long-form dialogue, multiple speakers, and overlapping speech better than any basic transcription service.',
  },
  {
    title: 'Marketing Agencies',
    description:
      'Delivering social media clips to clients? Typos look unprofessional. Run your VTT files through Clean Subtitle to ensure every brand name, product, and slogan is spelled correctly before you hit publish.',
  },
  {
    title: 'Translators & Localizers',
    description:
      'Starting with a clean English subtitle file is crucial for accurate translation. Use our tool to polish the source text before translating it into other languages, saving you hours of correction work down the line.',
  },
]

export function UseCases() {
  return (
    <section className="mt-24 rounded-4xl bg-neutral-950 py-24 sm:mt-32 lg:mt-40 lg:py-32">
      <Container>
        <FadeIn>
          <div className="max-w-2xl">
            <h2 className="font-display text-3xl font-medium tracking-tight text-white sm:text-4xl">
              Built for every creator.
            </h2>
            <p className="mt-6 text-lg tracking-tight text-neutral-400">
              Whether you are a solo YouTuber or a global media house, Clean Subtitle scales to meet your needs.
            </p>
          </div>
        </FadeIn>
        <div className="mt-16 grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {useCases.map((useCase) => (
            <FadeIn key={useCase.title} className="flex">
              <article className="relative flex w-full flex-col rounded-3xl p-6 ring-1 ring-neutral-800 transition hover:bg-neutral-900 sm:p-8">
                <h3>
                  <span className="absolute inset-0 rounded-3xl" />
                  <span className="font-display text-base font-semibold text-white">
                    {useCase.title}
                  </span>
                </h3>
                <p className="mt-4 text-base text-neutral-400">
                  {useCase.description}
                </p>
              </article>
            </FadeIn>
          ))}
        </div>
      </Container>
    </section>
  )
}
