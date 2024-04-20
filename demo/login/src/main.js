import Vue from 'vue'
import App from './App.vue'
import router from './router'

import '@/icons/index.js' 

import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'

import axios from 'axios'
// 创建axios实例
const axiosInstance = axios.create({
  baseURL: 'http://localhost:8080',  // 更正的Spring Boot后端地址
  timeout: 10000,
  headers: {
      'Content-Type': 'application/json'
  }
});

// 将axios实例绑定到Vue原型上，这样所有组件都可以通过this.$axios使用它
Vue.prototype.$axios = axiosInstance;


Vue.config.productionTip = false

Vue.use(ElementUI)


new Vue({
  render: h => h(App),
  router
}).$mount('#app')
