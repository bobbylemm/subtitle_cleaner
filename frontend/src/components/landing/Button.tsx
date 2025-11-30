import Link from 'next/link'
import clsx from 'clsx'

const baseStyles = {
  solid:
    'group inline-flex items-center justify-center rounded-full py-2 px-4 text-sm font-semibold focus-visible:outline-2 focus-visible:outline-offset-2',
  outline:
    'group inline-flex ring-1 items-center justify-center rounded-full py-2 px-4 text-sm',
}

const variantStyles = {
  solid: {
    slate:
      'bg-neutral-900 text-white hover:bg-neutral-700 hover:text-neutral-100 active:bg-neutral-800 active:text-neutral-300 focus-visible:outline-neutral-900',
    green:
      'bg-green-600 text-white hover:text-neutral-100 hover:bg-green-500 active:bg-green-800 active:text-green-100 focus-visible:outline-green-600',
    white:
      'bg-white text-neutral-900 hover:bg-green-50 active:bg-green-200 active:text-neutral-600 focus-visible:outline-white',
  },
  outline: {
    slate:
      'ring-neutral-200 text-neutral-700 hover:text-neutral-900 hover:ring-neutral-300 active:bg-neutral-100 active:text-neutral-600 focus-visible:outline-green-600 focus-visible:ring-neutral-300',
    white:
      'ring-neutral-700 text-white hover:ring-neutral-500 active:ring-neutral-700 active:text-neutral-400 focus-visible:outline-white',
  },
}

type ButtonProps = (
  | {
      variant?: 'solid'
      color?: keyof typeof variantStyles.solid
    }
  | {
      variant: 'outline'
      color?: keyof typeof variantStyles.outline
    }
) &
  (
    | Omit<React.ComponentPropsWithoutRef<typeof Link>, 'color'>
    | (Omit<React.ComponentPropsWithoutRef<'button'>, 'color'> & {
        href?: undefined
      })
  )

export function Button({ className, ...props }: ButtonProps) {
  props.variant ??= 'solid'
  props.color ??= 'slate'

  className = clsx(
    baseStyles[props.variant],
    props.variant === 'outline'
      ? variantStyles.outline[props.color]
      : props.variant === 'solid'
        ? variantStyles.solid[props.color]
        : undefined,
    className,
  )

  return typeof props.href === 'undefined' ? (
    <button className={className} {...props} />
  ) : (
    <Link className={className} {...props} />
  )
}
