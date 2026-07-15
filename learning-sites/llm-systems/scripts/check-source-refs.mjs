import { readFileSync, readdirSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const contentRoots = ['vllm', 'sglang', 'sft', 'distributed', 'verl']
const markdownFiles = contentRoots.flatMap((root) =>
  readdirSync(join(siteRoot, root), { recursive: true })
    .filter((entry) => entry.endsWith('.md'))
    .map((entry) => join(siteRoot, root, entry))
)

const failures = []
let checkedRefs = 0

for (const file of markdownFiles) {
  const source = readFileSync(file, 'utf8')
  for (const match of source.matchAll(/https:\/\/github\.com\/[^/)]+\/[^/)]+\/(?:blob|tree)\/([^/)#\s]+)/g)) {
    checkedRefs += 1
    if (!/^[0-9a-f]{40}$/.test(match[1])) {
      failures.push(`${relative(siteRoot, file)} -> ${match[0]}`)
    }
  }
}

if (failures.length > 0) {
  console.error(`Found ${failures.length} GitHub source link(s) without an exact 40-character commit:`)
  for (const failure of [...new Set(failures)].sort()) console.error(`- ${failure}`)
  process.exit(1)
}

console.log(`Checked ${checkedRefs} GitHub blob/tree references: every link uses an exact commit.`)
