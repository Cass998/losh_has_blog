import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import AcademyHome from './components/AcademyHome.vue'
import ImageLightbox from './components/ImageLightbox.vue'
import LessonMeta from './components/LessonMeta.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      'home-hero-before': () => h(AcademyHome),
      'doc-before': () => h(LessonMeta),
      'layout-bottom': () => h(ImageLightbox)
    })
}
