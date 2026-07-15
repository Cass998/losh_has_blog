<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'

const { frontmatter } = useData()
const lesson = computed(() => frontmatter.value.lesson)
</script>

<template>
  <header v-if="lesson" class="lesson-meta">
    <div class="lesson-meta__line">
      <span class="lesson-meta__stage">{{ lesson.track }}</span>
      <span>{{ lesson.stage }}</span>
      <span v-if="lesson.time">{{ lesson.time }}</span>
      <span v-if="lesson.level">{{ lesson.level }}</span>
    </div>
    <p v-if="lesson.question"><strong>这一课要回答：</strong>{{ lesson.question }}</p>
  </header>
</template>
