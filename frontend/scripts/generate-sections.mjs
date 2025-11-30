import glob from 'fast-glob'
import * as fs from 'fs'
import * as path from 'path'
import { toString } from 'mdast-util-to-string'
import { remark } from 'remark'
import remarkMdx from 'remark-mdx'
import { filter } from 'unist-util-filter'
import { SKIP, visit } from 'unist-util-visit'
import { slugifyWithCounter } from '@sindresorhus/slugify'

const processor = remark().use(remarkMdx).use(extractSections)
const slugify = slugifyWithCounter()

function isObjectExpression(node) {
    return (
        node.type === 'mdxTextExpression' &&
        node.data?.estree?.body?.[0]?.expression?.type === 'ObjectExpression'
    )
}

function excludeObjectExpressions(tree) {
    return filter(tree, (node) => !isObjectExpression(node))
}

function extractSections() {
    return (tree, { sections }) => {
        slugify.reset()

        visit(tree, (node) => {
            if (node.type === 'heading' || node.type === 'paragraph') {
                let content = toString(excludeObjectExpressions(node))
                if (node.type === 'heading' && node.depth === 2) {
                    let hash = slugify(content)
                    sections.push({
                        id: hash,
                        title: content,
                        offsetRem: 0,
                        tag: undefined,
                        headingRef: undefined
                    })
                }
                // We ignore paragraph content as it's not in the Section interface
                return SKIP
            }
        })
    }
}

async function main() {
    const cwd = path.join(process.cwd(), 'src/app')
    const pages = await glob('**/*.mdx', { cwd })

    const allSectionsEntries = await Promise.all(
        pages.map(async (filename) => {
            const filePath = path.join(cwd, filename)
            const mdx = fs.readFileSync(filePath, 'utf8')
            const sections = []
            const vfile = { value: mdx, sections }
            processor.runSync(processor.parse(vfile), vfile)

            return [
                '/' + filename.replace(/(^|\/)page\.mdx$/, ''),
                sections,
            ]
        })
    )

    const allSections = Object.fromEntries(allSectionsEntries)

    const outputPath = path.join(process.cwd(), 'src/generated/sections.json')
    fs.mkdirSync(path.dirname(outputPath), { recursive: true })
    fs.writeFileSync(outputPath, JSON.stringify(allSections, null, 2))

    console.log(`✅ Generated sections.json at ${outputPath}`)
}

main().catch(console.error)
