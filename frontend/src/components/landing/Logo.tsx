import clsx from 'clsx'

export function Logo(props: React.ComponentPropsWithoutRef<'div'>) {
  return (
    <div className={clsx("font-display font-bold text-xl text-zinc-900 dark:text-white", props.className)}>
      Clean Subtitle
    </div>
  )
}
