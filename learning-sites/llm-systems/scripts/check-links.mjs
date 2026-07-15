import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const contentRoots = ['vllm', 'sglang', 'sft', 'distributed', 'verl']

function markdownFiles(directory) {
  if (!existsSync(directory)) return []
  return readdirSync(directory, { recursive: true })
    .filter((entry) => entry.endsWith('.md'))
    .map((entry) => join(directory, entry))
}

function withoutFencedCode(markdown) {
  const kept = []
  let fence = null

  for (const line of markdown.split('\n')) {
    const marker = line.match(/^\s*(```+|~~~+)/)?.[1]
    if (marker && fence === null) {
      fence = marker[0]
      continue
    }
    if (marker && fence === marker[0]) {
      fence = null
      continue
    }
    if (fence === null) kept.push(line)
  }

  return kept.join('\n')
}

function linkTargets(markdown) {
  const content = withoutFencedCode(markdown)
  const targets = []

  for (const match of content.matchAll(/!?\[[^\]]*\]\(([^)\n]+)\)/g)) {
    const target = match[1].trim().match(/^<([^>]+)>|^(\S+)/)?.slice(1).find(Boolean)
    if (target) targets.push(target)
  }
  for (const match of content.matchAll(/^\s{0,3}\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))/gm)) {
    targets.push(match[1] || match[2])
  }
  for (const match of content.matchAll(/\b(?:href|src)=["']([^"']+)["']/g)) {
    targets.push(match[1])
  }

  return targets
}

function isExternal(target) {
  return /^(?:[a-z][a-z+.-]*:|\/\/|#)/i.test(target)
}

function targetExists(sourceFile, target) {
  const cleanTarget = target.split('#', 1)[0].split('?', 1)[0]
  if (!cleanTarget) return true

  const absolute = cleanTarget.startsWith('/')
    ? resolve(siteRoot, `.${cleanTarget}`)
    : resolve(dirname(sourceFile), cleanTarget)
  const candidates = [absolute, `${absolute}.md`, join(absolute, 'index.md')]

  if (cleanTarget.startsWith('/')) {
    candidates.push(resolve(siteRoot, 'public', `.${cleanTarget}`))
  }

  return candidates.some(existsSync)
}

const pages = [join(siteRoot, 'index.md'), ...contentRoots.flatMap((root) => markdownFiles(join(siteRoot, root)))]
const failures = []
let checkedLinks = 0

for (const page of pages) {
  for (const target of linkTargets(readFileSync(page, 'utf8'))) {
    if (isExternal(target)) continue
    checkedLinks += 1
    if (!targetExists(page, target)) {
      failures.push(`${relative(siteRoot, page)} -> ${target}`)
    }
  }
}

const navigationFiles = [
  join(siteRoot, '.vitepress', 'config.mts'),
  join(siteRoot, '.vitepress', 'theme', 'components', 'AcademyHome.vue')
]

for (const file of navigationFiles) {
  const source = readFileSync(file, 'utf8')
  for (const match of source.matchAll(/\b(?:link|href):?\s*["'](\/[^"']+)["']/g)) {
    checkedLinks += 1
    if (!targetExists(file, match[1])) {
      failures.push(`${relative(siteRoot, file)} -> ${match[1]}`)
    }
  }
}

if (failures.length > 0) {
  console.error(`Found ${failures.length} broken local link(s):`)
  for (const failure of [...new Set(failures)].sort()) console.error(`- ${failure}`)
  process.exit(1)
}

console.log(`Checked ${checkedLinks} local links across ${pages.length} pages: all targets exist.`)
