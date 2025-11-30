import Image from 'next/image'

import { Container } from '@/components/landing/Container'
import backgroundImage from '@/images/landing/background-faqs.jpg'

const faqs = [
  [
    {
      question: 'Does Clean Subtitle work with VTT files?',
      answer:
        'Yes, we support both SRT and VTT formats. The output will maintain the original format.',
    },
    {
      question: 'Is my data secure?',
      answer: 'Absolutely. We process your files in a secure environment and delete them immediately after processing. We do not use your data to train our models.',
    },
    {
      question: 'How accurate is the correction?',
      answer:
        'Our system uses advanced LLMs with a specialized context-aware engine. It is significantly more accurate than standard spell checkers or basic AI prompts.',
    },
  ],
  [
    {
      question: 'Can I customize the style guide?',
      answer:
        'Yes, you can provide specific instructions or a style guide to ensure the corrections match your brand voice.',
    },
    {
      question:
        'What languages are supported?',
      answer:
        'Currently, we focus on English (US/UK), but our engine is capable of handling multiple languages. Contact us for specific language requests.',
    },
    {
      question:
        'Do you offer an API?',
      answer:
        'Yes, we have a robust API for enterprise integration. Please contact our sales team for documentation and access.',
    },
  ],
  [
    {
      question: 'How does the pricing work?',
      answer:
        'We offer a free tier for individuals and paid plans for professionals and teams. Check our pricing section for details.',
    },
    {
      question: 'Can I cancel my subscription?',
      answer: 'Yes, you can cancel at any time. No questions asked.',
    },
    {
      question: 'Do you offer refunds?',
      answer:
        'If you are not satisfied with the results, please contact support within 14 days for a full refund.',
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
