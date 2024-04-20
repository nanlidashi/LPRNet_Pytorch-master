<template>
  <div class="login" :style="'background-image:url(' + Background + ');'">
    <el-form
      ref="loginForm"
      :model="loginForm"
      :rules="loginRules"
      label-position="left"
      label-width="0px"
      class="login-form"
    >
      <h3 class="title">欢迎使用</h3>
      <el-form-item prop="username">
        <el-input
          v-model="loginForm.username"
          type="text"
          auto-complete="off"
          placeholder="账号"
        >
          <svg-icon
            slot="prefix"
            icon-class="user"
            class="el-input__icon input-icon"
          />
        </el-input>
      </el-form-item>
      <el-form-item prop="password">
        <el-input
          v-model="loginForm.password"
          type="password"
          auto-complete="off"
          placeholder="密码"
          @keyup.enter.native="handleLogin"
        >
          <svg-icon
            slot="prefix"
            icon-class="password"
            class="el-input__icon input-icon"
          />
        </el-input>
      </el-form-item>
      <el-form-item prop="code">
        <el-input
          v-model="loginForm.code"
          auto-complete="off"
          placeholder="验证码"
          style="float:left;width: 66%"
          @keyup.enter.native="handleLogin"
        >
          <svg-icon
            slot="prefix"
            icon-class="validCode"
            class="el-input__icon input-icon"
          />
        </el-input>
        <div class="login-code" @click="getCode">
          <valid-code :identifyCode="identifyCode" />
        </div>
      </el-form-item>
      <el-checkbox v-model="loginForm.rememberMe" style="float:left;margin: 0 0 25px 0">
        记住我
      </el-checkbox>
      <el-form-item style="width: 100%">
        <el-button
          :loading="loading"
          size="medium"
          type="primary"
          style="width: 100%"
          @click.native.prevent="handleLogin"
        >
          <span v-if="!loading">登 录</span>
          <span v-else>登 录 中...</span>
        </el-button>
      </el-form-item>
      <p style="float:right;font-size: 13px">
        还没有帐号？
        <a href="/register" class="register">注册</a>
      </p>
    </el-form>
  </div>
</template>

<script>
import { encrypt } from '@/utils/rsaEncrypt'
import Cookies from 'js-cookie'
import validCode from '../components/validCode.vue'
import Background from '@/assets/images/top-scm.jpg'

export default {
  components: { validCode },
  name: 'Login',
  data() {
    return {
      Background: Background,
      identifyCodes: '1234567890abcdefjhijk1234567890linopqrsduvwxyz',
      identifyCode: '',
      cookiePass: '',
      loginForm: {
        username: '',
        password: '',
        rememberMe: false,
        code: '',
        uuid: ''
      },
      loginRules: {
        username: [{ required: true, trigger: 'blur', message: '用户名不能为空' }],
        password: [{ required: true, trigger: 'blur', message: '密码不能为空' }],
        code: [{ required: true, trigger: 'change', message: '验证码不能为空' }]
      },
      loading: false,
      redirect: undefined
    }
  },
  watch: {
    $route: {
      handler: function(route) {
        this.redirect = route.query && route.query.redirect
      },
      immediate: true
    }
  },
  created() {
    this.getCode()
    this.getCookie()
    this.point()
  },
  methods: {
    getCode() {
      this.identifyCode = ''
      for (let i = 0; i < 4; i++) {
        this.identifyCode += this.identifyCodes[this.randomNum(0, this.identifyCodes.length)]
      }
      console.log(this.identifyCode)
    },
    randomNum(min, max) {
      return Math.floor(Math.random() * (max - min) + min)
    },
    getCookie() {
      const username = Cookies.get('username')
      let password = Cookies.get('password')
      const rememberMe = Cookies.get('rememberMe')
      this.cookiePass = password === undefined ? '' : password
      password = password === undefined ? this.loginForm.password : password
      this.loginForm = {
        username: username === undefined ? this.loginForm.username : username,
        password: password,
        rememberMe: rememberMe === undefined ? false : Boolean(rememberMe),
        code: ''
      }
    },
    async handleLogin() {
      this.$refs.loginForm.validate(async (valid) => {
        const user = {
          username: this.loginForm.username,
          password: this.loginForm.password,
          rememberMe: this.loginForm.rememberMe,
          code: this.loginForm.code,
          uuid: this.loginForm.uuid
        }
        if (user.password !== this.cookiePass) {
          user.password = encrypt(user.password)
        }
        if (valid) {
          this.loading = true
          if (user.rememberMe) {
            Cookies.set('username', user.username, { expires: 1 })
            Cookies.set('password', user.password, { expires: 1 })
            Cookies.set('rememberMe', user.rememberMe, { expires: 1 })
          } else {
            Cookies.remove('username')
            Cookies.remove('password')
            Cookies.remove('rememberMe')
          }
          try {
            const response = await this.$axios.post('/login', user)
            if (response.status === 200) {
              this.$message({
                showClose: true,
                message: '登录成功，正在跳转到主页...',
                type: 'success'
              })
              setTimeout(() => {
                this.loading = false
                window.location.href = "/index"
              }, 2000)
            } else {
              this.$message({
                showClose: true,
                message: response.data || '登录失败',
                type: 'error'
              })
              this.loading = false
            }
          } catch (error) {
            console.error('登录失败:', error)
            this.$message({
              showClose: true,
              message: '登录失败，请稍后重试',
              type: 'error'
            })
            this.loading = false
          }
        } else {
          console.log('error submit!!')
          return false
        }
      })
    },
    point() {
      const point = Cookies.get('point') !== undefined
      if (point) {
        this.$notify({
          title: '提示',
          message: '当前登录状态已过期，请重新登录！',
          type: 'warning',
          duration: 5000
        })
        Cookies.remove('point')
      }
    }
  }
}
</script>

<style lang="scss" scoped src="../assets/css/login.scss">

</style>
