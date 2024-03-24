<template>
  <div>
    <el-container class="main-container">
      <el-header class="top-div">
        <p>车牌识别系统</p>
      </el-header>
      <el-container class="bottom-div">
        <el-aside class="left-side">
          <el-menu default-active="2" class="el-menu-vertical-demo" background-color="#545c64" text-color="#fff" active-text-color="#ffd04b">
            <el-menu-item index="1">
              <a href="/index">首页</a>
            </el-menu-item>
            <el-menu-item index="2">
              <a href="/upload">上传测试车牌</a>
            </el-menu-item>
            <el-menu-item index="3">
              <a href="/ceshi">历史测试车牌</a>
            </el-menu-item>
          </el-menu>
        </el-aside>
        <el-main class="right-side">
          <el-form ref="form" :model="formData" label-width="80px">
            <el-upload
              class="file-input"
              action="#"
              :on-change="handleChange"
              :auto-upload="false">
              <el-button slot="trigger" size="small" type="primary">选取文件</el-button>
            </el-upload>
            <div class="image-preview">
              <img :src="imageUrl" alt="Image Preview" v-if="imageUrl !== ''">
            </div>
            <el-button size="small" type="success" @click="submitUpload">识别</el-button>
            <p class="output">{{ uploadOutput }}</p>
          </el-form>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script>
export default {
  name: 'UpLoad',
  data() {
    return {
      formData: {},
      imageUrl: '',
      selectedFile: null,
      uploadOutput: ''
    };
  },
  methods: {
    handleChange(file) {
      this.selectedFile = file.raw;
      this.selectedFileName = '选择的文件：' + file.raw.name;
      const reader = new FileReader();
      reader.onload = (event) => {
        this.imageUrl = event.target.result;
      };
      reader.readAsDataURL(file.raw);
    },
    async submitUpload() {
      try {
        const formData = new FormData();
        formData.append('file', this.selectedFile);

        const response = await this.$axios.post('/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });

        this.uploadOutput = response.data;
      } catch (error) {
        console.error('Error uploading file:', error);
        this.uploadOutput = '上传失败';
      }
    }
  }
};
</script>

<style>
body, html {
  margin: 0;
  padding: 0;
  height: 100%;
  background-color: blue;
}

.main-container {
  height: 100vh;
  display: grid;
  grid-template-rows: 20% 80%;
}

.top-div {
  background-color: lightblue;
  border: 1px solid #ccc;
  display: flex;
  justify-content: center;
  align-items: center;
}

.top-div p {
  margin: 0;
}

.bottom-div {
  display: grid;
  grid-template-columns: 1fr 1fr;
}

.left-side {
  background-color: lightgray;
  border: 1px solid #ccc;
  width: 300px;
  padding: 10px;
}

.left-side ul {
  list-style-type: none;
  padding: 0;
  justify-content: center;
  flex-direction: column;
}

.left-side li {
  font-size: 18px;
  margin-bottom: 15px;
  cursor: pointer;
  transition: 0.3s;
  text-align: center;
}

.left-side li a {
  text-decoration: none;
  color: inherit;
}

.left-side li:hover {
  background-color: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
}

.right-side {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
  background-color: lightgray;
  border: 1px solid #ccc;
  width: 1000px;
}

.file-input {
  margin-bottom: 20px;
}

.image-preview {
  width: 300px;
  height: 300px;
  border: 2px solid #ccc;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f0f0f0;
  margin-bottom: 20px;
}

.image-preview img {
  max-width: 100%;
  max-height: 100%;
}

.selected-file-path {
  margin-bottom: 20px;
  font-size: 16px;
}

.output {
  font-size: 16px;
}
</style>
