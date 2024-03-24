<template>
  <body id="poster">
    <el-form class="login-container" label-position="left"
             label-width="0px">
      <h3 class="login_title">车牌识别系统登录</h3>
      <el-form-item>
        <el-input type="text" v-model="loginForm.username"
                  auto-complete="off" placeholder="账号" class="input-transparent"></el-input>
      </el-form-item>
      <el-form-item>
        <el-input type="password" v-model="loginForm.password"
                  auto-complete="off" placeholder="密码" class="input-transparent"></el-input>
      </el-form-item>
      <el-form-item style="width: 100%">
        <el-button type="primary" style="width: 100%;background: #000;border: none" v-on:click="login">登录</el-button>
      </el-form-item>
    </el-form>
  </body>
</template>

<script>

export default {
  name: 'DeLu',
  data () {
    return {
      loginForm: {
        username: 'admin',
        password: '123456'
      },
      responseResult: []
    }
  },
  methods: {
    async login () {
        try {
            let response = await this.$axios.post('/login', {
                username: this.loginForm.username,
                password: this.loginForm.password
            });

            console.log(response.data);
            
            if (response.data.code === 200) {
                console.log("跳转到index页面");
                this.$router.push({ path: '/index' });
            } else {
                alert(response.data.message || '登录失败');
            }
        } catch (error) {
            console.error('登录失败:', error);
            alert('登录失败，请稍后重试');
        }
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
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
.login-container {
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
.login_title {
  margin: 0px auto 40px auto;
  text-align: center;
  color: #1d1e1f;
}
.input-transparent {
  color: #000;
}
</style>
