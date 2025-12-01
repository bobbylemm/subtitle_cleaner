import { Container } from '@/components/landing/Container'

const benefits = [
  {
    title: 'Save Hours of Editing Time',
    description:
      'Manual subtitle correction is tedious and slow. Our AI processes an hour-long video in seconds, freeing you up to focus on creative work.',
  },
  {
    title: 'Boost Audience Retention',
    description:
      'Errors in subtitles break immersion and cause viewers to click away. Perfect grammar and spelling keep your audience engaged till the end.',
  },
  {
    title: 'Professional Quality for Everyone',
    description:
      'Whether you are a solo creator or a large media house, deliver broadcast-quality captions that respect your brand voice and style guide.',
  },
  {
    title: 'Global Reach, Local Feel',
    description:
      'Our context-aware engine understands regional slang and idioms, ensuring your message resonates authentically with viewers worldwide.',
  },
  {
    title: 'Seamless Workflow Integration',
    description:
      'Export compatible .srt and .vtt files that drop right into Premiere Pro, DaVinci Resolve, Final Cut, or YouTube Studio without formatting issues.',
  },
  {
    title: 'Data Privacy First',
    description:
      'We do not train our models on your private content. Your files are processed in an isolated environment and automatically deleted after 24 hours.',
  },
]

export function Benefits() {
  return (
    <section
      id="benefits"
      aria-label="Benefits of using Clean Subtitle"
      className="bg-white py-20 sm:py-32"
    >
      <Container>
        <div className="mx-auto max-w-2xl md:text-center">
          <h2 className="font-display text-3xl tracking-tight text-neutral-900 sm:text-4xl">
            Why Creators Choose Clean Subtitle
          </h2>
          <p className="mt-4 text-lg tracking-tight text-neutral-700">
            Subtitle accuracy isn't just about being correct; it's about respect for your content and your audience.
          </p>
        </div>
        <ul
          role="list"
          className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-6 sm:gap-8 lg:mt-20 lg:max-w-none lg:grid-cols-3"
        >
          {benefits.map((benefit) => (
            <li
              key={benefit.title}
              className="rounded-2xl border border-neutral-200 p-8 hover:bg-neutral-50 transition-colors"
            >
              <h3 className="font-display text-lg font-semibold text-neutral-900">
                {benefit.title}
              </h3>
              <p className="mt-4 text-sm text-neutral-600 leading-6">
                {benefit.description}
              </p>
            </li>
          ))}
        </ul>
      </Container>
    </section>
  )
}
