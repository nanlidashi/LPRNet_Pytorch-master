<template>
  <body id="poster">
    <el-form ref="loginFormRef" :model="loginForm" class="login-container-1" label-position="left" label-width="0px">
      <h3 class="login_title-1">车牌识别系统注册</h3>
      <el-form-item prop="username">
        <el-input type="text" v-model="loginForm.username" auto-complete="off" placeholder="账号" class="input-transparent-1"></el-input>
      </el-form-item>
      <el-form-item  prop="password">
        <el-input type="password" v-model="loginForm.password" auto-complete="off" placeholder="输入密码" class="input-transparent-1"></el-input>
      </el-form-item>
      <el-form-item  prop="confirmPassword">
        <el-input type="password" v-model="loginForm.confirmPassword" auto-complete="off" placeholder="请再次输入密码" class="input-transparent-1"></el-input>
      </el-form-item>
      <el-form-item style="width: 100%">
        <el-button type="primary" style="width: 100%;background: #000;border: none; margin-top: 10px;" v-on:click="register">注册</el-button>
      </el-form-item>
      <el-form-item style="width: 100%">
        <el-button type="primary" style="width: 100%;background: #000;border: none; margin-top: 10px;" v-on:click="goToLogin">已有账号？去登录</el-button>
      </el-form-item>
    </el-form>
  </body>
</template>

<script>
export default {
  name: 'ZhuCe',
  data () {
    return {
      loginForm: {
        username: '',
        password: '',
        confirmPassword: ''
      },
    }
  },
  methods: {
    async register() {
      try {
        let response = await this.$axios.post('/register', {
          username: this.loginForm.username,
          password: this.loginForm.password,
          confirmPassword: this.loginForm.confirmPassword
        });

        console.log(response.data);

        if (response.data === "用户名已存在" || response.data === "密码不一致") {
          alert(response.data);
        } else {
          console.log("跳转到登录页面");
          this.$router.push({ path: '/login' });
        }
      } catch (error) {
        console.error('注册失败:', error);
        alert('注册失败，请稍后重试');
      }
    },
    goToLogin() {
      this.$router.push({ path: '/login' });
    }
  }
}
</script>


<style scoped>
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  width: 100%;
  overflow: hidden;
}

#poster {
  background: url("../assets/funingna.jpg") no-repeat;
  background-position: center;
  height: 100vh;
  width: 100%;
  background-size: cover;
  position: fixed;
  top: 0;
  left: 0;
}

.login-container-1 {
  border-radius: 15px;
  background-clip: padding-box;
  margin: 90px auto;
  width: 350px;
  padding: 35px 35px 15px 35px;
  background: #fff;
  border: 1px solid #eaeaea;
  box-shadow: 0 0 25px #cac6c6;
  opacity: 0.9;
}

.login_title-1 {
  margin: 0px auto 40px auto;
  text-align: center;
  color: #1d1e1f;
}

.input-transparent-1 {
  color: #000;
}
</style>
