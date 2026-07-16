<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

const selector = '.vp-doc img, .vp-doc .mermaid svg'
const minScale = 0.5
const maxScale = 6

const dialog = ref<HTMLDialogElement>()
const closeButton = ref<HTMLButtonElement>()
const source = ref('')
const caption = ref('课程图表')
const scale = ref(1)
const offsetX = ref(0)
const offsetY = ref(0)

let blobUrl = ''
let previousFocus: HTMLElement | null = null
let observer: MutationObserver | undefined

function targetFrom(value: EventTarget | null) {
  return value instanceof Element ? value.closest<HTMLElement>(selector) : null
}

function decorateImages() {
  document.querySelectorAll<HTMLElement>(selector).forEach((image) => {
    if (image.dataset.lightboxReady) return
    const description = image instanceof HTMLImageElement
      ? image.alt || '课程图片'
      : image.getAttribute('aria-label') || '课程图表'

    image.dataset.lightboxReady = 'true'
    image.tabIndex = 0
    image.setAttribute('aria-haspopup', 'dialog')
    image.setAttribute('aria-label', `${description}，点击或按回车放大`)
    image.title = '点击放大'
  })
}

function resetZoom() {
  scale.value = 1
  offsetX.value = 0
  offsetY.value = 0
}

function setScale(next: number, anchorX = 0, anchorY = 0) {
  const bounded = Math.min(maxScale, Math.max(minScale, next))
  const ratio = bounded / scale.value
  offsetX.value = anchorX - (anchorX - offsetX.value) * ratio
  offsetY.value = anchorY - (anchorY - offsetY.value) * ratio
  scale.value = bounded
}

function openImage(target: HTMLElement) {
  if (blobUrl) URL.revokeObjectURL(blobUrl)

  if (target instanceof HTMLImageElement) {
    source.value = target.currentSrc || target.src
    caption.value = target.alt || '课程图片'
  } else {
    const svg = target.cloneNode(true) as SVGElement
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    blobUrl = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], { type: 'image/svg+xml' }))
    source.value = blobUrl
    caption.value = target.getAttribute('aria-label')?.replace('，点击或按回车放大', '') || '课程图表'
  }

  resetZoom()
  previousFocus = document.activeElement as HTMLElement | null
  document.body.classList.add('lightbox-open')
  dialog.value?.showModal()
  nextTick(() => closeButton.value?.focus())
}

function close() {
  dialog.value?.close()
}

function onClosed() {
  document.body.classList.remove('lightbox-open')
  if (blobUrl) URL.revokeObjectURL(blobUrl)
  blobUrl = ''
  source.value = ''
  previousFocus?.focus()
}

function onDocumentClick(event: MouseEvent) {
  const target = targetFrom(event.target)
  if (!target) return
  event.preventDefault()
  openImage(target)
}

function onDocumentKeydown(event: KeyboardEvent) {
  const target = targetFrom(event.target)
  if (!target || (event.key !== 'Enter' && event.key !== ' ')) return
  event.preventDefault()
  openImage(target)
}

function onWheel(event: WheelEvent) {
  const stage = event.currentTarget as HTMLElement
  const rect = stage.getBoundingClientRect()
  const anchorX = event.clientX - rect.left - rect.width / 2
  const anchorY = event.clientY - rect.top - rect.height / 2
  setScale(scale.value * (event.deltaY < 0 ? 1.15 : 1 / 1.15), anchorX, anchorY)
}

function onStageClick(event: MouseEvent) {
  if (event.target === event.currentTarget) close()
}

onMounted(() => {
  decorateImages()
  document.addEventListener('click', onDocumentClick)
  document.addEventListener('keydown', onDocumentKeydown)
  observer = new MutationObserver(decorateImages)
  observer.observe(document.body, { childList: true, subtree: true })
})

onBeforeUnmount(() => {
  observer?.disconnect()
  document.removeEventListener('click', onDocumentClick)
  document.removeEventListener('keydown', onDocumentKeydown)
  document.body.classList.remove('lightbox-open')
  if (blobUrl) URL.revokeObjectURL(blobUrl)
})
</script>

<template>
  <dialog ref="dialog" class="image-lightbox" aria-label="图片预览" @close="onClosed">
    <div class="image-lightbox__toolbar">
      <button type="button" aria-label="缩小图片" title="缩小" @click="setScale(scale / 1.25)">−</button>
      <button type="button" aria-label="恢复原始比例" title="恢复原始比例" @click="resetZoom">
        {{ Math.round(scale * 100) }}%
      </button>
      <button type="button" aria-label="放大图片" title="放大" @click="setScale(scale * 1.25)">+</button>
      <button ref="closeButton" type="button" aria-label="关闭图片预览" title="关闭" @click="close">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" />
        </svg>
      </button>
    </div>

    <div class="image-lightbox__stage" @click="onStageClick" @wheel.prevent="onWheel">
      <img
        v-if="source"
        class="image-lightbox__media"
        :src="source"
        :alt="caption"
        :style="{ transform: `translate3d(${offsetX}px, ${offsetY}px, 0) scale(${scale})` }"
      >
    </div>

    <p class="image-lightbox__hint">滚轮缩放 · Esc 关闭</p>
  </dialog>
</template>
