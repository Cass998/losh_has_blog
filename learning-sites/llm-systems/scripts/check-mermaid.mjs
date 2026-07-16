import fs from 'node:fs'
import path from 'node:path'
import DOMPurify from 'dompurify'

if (typeof DOMPurify.addHook !== 'function') {
  DOMPurify.addHook = () => {}
  DOMPurify.sanitize = (value) => value
}

const { default: mermaid } = await import('mermaid')
const roots = ['foundations', 'vllm', 'sglang', 'sft', 'distributed', 'verl']
const files = []

function collect(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name)
    if (entry.isDirectory()) collect(target)
    else if (entry.name.endsWith('.md')) files.push(target)
  }
}

for (const root of roots) {
  if (fs.existsSync(root)) collect(root)
}

let count = 0
const failures = []

for (const file of files) {
  const markdown = fs.readFileSync(file, 'utf8')
  let localCount = 0
  for (const match of markdown.matchAll(/```mermaid\n([\s\S]*?)```/g)) {
    count += 1
    localCount += 1
    try {
      await mermaid.parse(match[1].replace(/<br\s*\/?\s*>/gi, ' — '))
    } catch (error) {
      failures.push(`${file} #${localCount}: ${error.message}`)
    }
  }
}

if (failures.length) {
  console.error(failures.join('\n'))
  process.exit(1)
}

console.log(`Validated ${count} Mermaid diagrams in ${files.length} lessons.`)
