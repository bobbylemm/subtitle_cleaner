import Link from 'next/link'

import clsx from 'clsx'

export function NavLink({
  href,
  children,
  className,
}: {
  href: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <Link
      href={href}
      className={clsx(
        'inline-block rounded-lg px-2 py-1 text-sm hover:bg-neutral-100 hover:text-neutral-900',
        className ?? 'text-neutral-700',
      )}
    >
      {children}
    </Link>
  )
}
