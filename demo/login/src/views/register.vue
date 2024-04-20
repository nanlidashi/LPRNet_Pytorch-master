<template>
  <div class="page-register">
    <article class="header">
      <header>
        <el-avatar icon="el-icon-user-solid" shape="circle"></el-avatar>
        <span class="login">
          <em class="bold">已有账号？</em>
          <a href="/login">
            <el-button type="primary" size="small">登录</el-button>
          </a>
        </span>
      </header>
    </article>
    <el-steps :active="active" finish-status="success">
      <el-step title="步骤 1"></el-step>
      <el-step title="步骤 2"></el-step>
    </el-steps>

    <section>
      <el-form
        ref="ruleForm"
        :model="ruleForm"
        :rules="rules"
        label-width="100px"
        autocomplete="off"
        size="medium"
      >
        <div v-if="active == 0">
          <el-form-item prop="textarea">
            <el-input
              :value="ruleForm.textarea"
              type="textarea"
              :rows="10"
              :readonly="true"
            >
            </el-input>
          </el-form-item>
          <el-form-item prop="agreed">
            <el-checkbox v-model="ruleForm.agreed" style="float: left"
              >同意注册协议</el-checkbox
            >
          </el-form-item>
        </div>
        <div v-if="active == 1">
          <el-form-item label="用户名" prop="name">
            <el-input v-model="ruleForm.name" />
          </el-form-item>
          <el-form-item label="邮箱" prop="email">
            <el-input v-model="ruleForm.email" />
            <el-button
              size="mini"
              round
              @click="sendMsg"
              style="margin-top: 2px; float: left"
              >发送验证码</el-button
            >
            <span class="status">{{ statusMsg }}</span>
          </el-form-item>
          <el-form-item label="验证码" prop="code">
            <el-input v-model="ruleForm.code" maxlength="4" />
          </el-form-item>
          <el-form-item label="密码" prop="password">
            <el-input v-model="ruleForm.password" type="password" />
          </el-form-item>
          <el-form-item label="确认密码" prop="cpassword">
            <el-input v-model="ruleForm.cpassword" type="password" />
          </el-form-item> 
        </div>
      </el-form>
    </section>
    <div class="footer">
      <el-button
        v-if="active > 0"
        type="primary"
        icon="el-icon-arrow-left"
        @click="prev"
        >上一步</el-button
      >
      <el-button
        v-if="active < step - 1"
        type="primary"
        icon="el-icon-arrow-right"
        @click="next"
        >下一步</el-button
      >
      <el-button v-if="active == step - 1" type="primary" @click="register"
        >同意以下协议并注册</el-button
      >
      <div class="error">{{ error }}</div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'Register',
  data() {
    return {
      step: 2,
      active: 0,
      statusMsg: '',
      error: '',
      ruleForm: {
        textarea: '请仔细阅读以下协议',
        agreed: false,
        name: '',
        code: '',
        password: '',
        cpassword: '',
        email: ''
      },
      rules: {
        agreed: [{
          validator: (rule, value, callback) => {
            if (value !== true) {
              callback(new Error('请确认同意注册协议'))
            } else {
              callback()
            }
          },
          trigger: 'blur'
        }],
        name: [{
          required: true,
          type: 'string',
          message: '请输入用户名',
          trigger: 'blur'
        }],
        email: [{
          required: true,
          type: 'email',
          message: '请输入邮箱',
          trigger: 'blur'
        }],
        password: [{
          required: true,
          message: '创建密码',
          trigger: 'blur'
        }],
        cpassword: [{
          required: true,
          message: '确认密码',
          trigger: 'blur'
        }]
      }
    }
  },
  layout: 'blank',
  methods: {
    sendMsg: function() {
      const self = this
      let namePass
      let emailPass
      if (self.timerid) {
        return false
      }
      this.$refs['ruleForm'].validateField('name', (valid) => {
        namePass = valid
      })
      self.statusMsg = ''
      if (namePass) {
        return false
      }
      this.$refs['ruleForm'].validateField('email', (valid) => {
        emailPass = valid
      })
      // 模拟验证码发送
      if (!namePass && !emailPass) {
        let count = 60
        self.statusMsg = `验证码已发送,剩余${count--}秒`
        self.timerid = setInterval(function() {
          self.statusMsg = `验证码已发送,剩余${count--}秒`
          if (count === 0) {
            clearInterval(self.timerid)
          }
        }, 1000)
      }
    },

    next: function() {
      if (this.active === 0) {
        this.$refs['ruleForm'].validateField('agreed', (valid) => {
          if (valid === '') {
            this.active++
          }
        })
      }
    },

    prev: function() {
      this.$refs['ruleForm'].clearValidate()
      if (--this.active < 0) this.active = 0
    },

    register: async function() {
      this.$refs['ruleForm'].validate(async (valid) => {
        if (valid) {
          try {
            const response = await axios.post('http://localhost:8080/api/register', {
              username: this.ruleForm.name,
              email: this.ruleForm.email,
              password: this.ruleForm.password,
              cPassword: this.ruleForm.cpassword
            })
            if (response.status === 200) {
              this.$message({
                showClose: true,
                message: '注册成功，正在跳转到登录界面...',
                type: 'success'
              })
              setTimeout(() => {
                this.$router.push('/login')
              }, 2000)
            } else {
              this.error = response.data.message || '注册失败'
            }
          } catch (error) {
            console.error('注册失败:', error)
            this.$message({
              showClose: true,
              message: '注册失败，请稍后重试',
              type: 'error'
            })
          }
        }
      })
    }
  }
}
</script>

<style scoped lang="scss" src="../assets/css/register.scss">

</style>
