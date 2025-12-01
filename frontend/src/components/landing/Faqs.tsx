import Image from 'next/image'

import { Container } from '@/components/landing/Container'
import backgroundImage from '@/images/landing/background-faqs.jpg'

const faqs = [
  [
    {
      question: 'Does Clean Subtitle work with VTT files?',
      answer:
        'Yes, we fully support both SubRip (.srt) and WebVTT (.vtt) formats. Whether you are exporting from Adobe Premiere Pro, DaVinci Resolve, or Final Cut Pro, our system preserves your original timestamps and formatting while correcting the text. You can seamlessly switch between formats if needed.',
    },
    {
      question: 'Is my data secure?',
      answer: 'Absolutely. Security is our top priority. We process your subtitle files in an isolated, encrypted environment. Once the correction process is complete and you have downloaded your file, the data is permanently deleted from our servers. We do not use your content to train our public AI models.',
    },
    {
      question: 'How accurate is the correction?',
      answer:
        'Our system uses advanced Large Language Models (LLMs) combined with a specialized context-aware engine. Unlike basic spell checkers that only look at individual words, Clean Subtitle analyzes the entire sentence and surrounding context. This allows it to fix complex issues like homophones (e.g., "their" vs "there"), proper nouns (e.g., "Mbappé"), and technical jargon with over 98% accuracy.',
    },
    {
      question: 'Can it fix synchronization issues?',
      answer:
        'While our primary focus is on text correction (grammar, spelling, punctuation), we also perform basic structural validation. If your file has overlapping timestamps or invalid formatting that might cause playback errors on YouTube or VLC, our system will identify and often auto-correct these structural problems.',
    },
  ],
  [
    {
      question: 'Can I customize the style guide?',
      answer:
        'Yes! We understand that every creator has a unique voice. You can provide specific instructions, such as "Use US English spelling," "Always capitalize specific terms," or "Keep slang words like \'gonna\'." Our AI adapts to your preferences, ensuring the final subtitles match your brand identity perfectly.',
    },
    {
      question: 'What languages are supported?',
      answer:
        'Currently, we specialize in English (US, UK, Australian, etc.), including various dialects and accents. Our engine is particularly good at handling non-native English speakers and heavy accents by using phonetic context. We are actively working on adding support for Spanish, French, and German in the near future.',
    },
    {
      question: 'Do you offer an API for developers?',
      answer:
        'Yes, we offer a robust REST API for enterprise integration. If you are a media company, streaming platform, or post-production house looking to automate subtitle correction at scale, our API provides all the features of the web tool programmatically. Contact our sales team for documentation and API keys.',
    },
    {
      question: 'How does it handle slang and informal speech?',
      answer:
        'One of the biggest challenges with standard auto-correct is that it often "fixes" intentional slang, ruining the vibe of the content. Clean Subtitle is designed to recognize and preserve intentional slang, idioms, and informal speech patterns, ensuring your subtitles feel authentic to the speaker\'s voice.',
    },
  ],
  [
    {
      question: 'How does the pricing work?',
      answer:
        'We offer a generous free tier that allows you to process a limited number of files per day, perfect for individual creators. For professionals and teams requiring higher volume, faster processing speeds, and priority support, we offer affordable monthly and annual subscription plans. Check our pricing section for full details.',
    },
    {
      question: 'Can I cancel my subscription?',
      answer: 'Yes, you can cancel your subscription at any time directly from your dashboard. There are no long-term contracts or hidden fees. If you cancel, you will retain access to your paid features until the end of your current billing cycle.',
    },
    {
      question: 'Do you offer refunds?',
      answer:
        'We stand by the quality of our service. If you are not completely satisfied with the results, please contact our support team within 14 days of your purchase. We will work with you to resolve the issue or provide a full refund, no questions asked.',
    },
    {
      question: 'What if I have a very large file?',
      answer:
        'Our system is optimized to handle feature-length movies and long-form podcasts. We support files up to 50MB in size, which covers almost any subtitle file you will encounter. For extremely large batch processing needs, please reach out to our enterprise team.',
    },
  ],
]

export function Faqs() {
  return (
    <section
      id="faq"
      aria-labelledby="faq-title"
      className="relative overflow-hidden bg-neutral-50 py-20 sm:py-32"
    >
      <Image
        className="absolute top-0 left-1/2 max-w-none translate-x-[-30%] -translate-y-1/4"
        src={backgroundImage}
        alt=""
        width={1558}
        height={946}
        unoptimized
      />
      <Container className="relative">
        <div className="mx-auto max-w-2xl lg:mx-0">
          <h2
            id="faq-title"
            className="font-display text-3xl tracking-tight text-neutral-900 sm:text-4xl"
          >
            Frequently asked questions
          </h2>
          <p className="mt-4 text-lg tracking-tight text-neutral-700">
            If you can’t find what you’re looking for, email our support team
            and if you’re lucky someone will get back to you.
          </p>
        </div>
        <ul
          role="list"
          className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-8 lg:max-w-none lg:grid-cols-3"
        >
          {faqs.map((column, columnIndex) => (
            <li key={columnIndex}>
              <ul role="list" className="flex flex-col gap-y-8">
                {column.map((faq, faqIndex) => (
                  <li key={faqIndex}>
                    <h3 className="font-display text-lg/7 text-neutral-900">
                      {faq.question}
                    </h3>
                    <p className="mt-4 text-sm text-neutral-700">{faq.answer}</p>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      </Container>
    </section>
  )
}
