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
        <div class="right-side-upload">
          <div class="img">
            <div class="left-img">
              <!-- 文件选择和上传 -->
              <div class="left-file-input">
                <input type="file" @change="previewImage" ref="fileInput" style="display: none;">
                <el-button type="primary" plain @click="$refs.fileInput.click()">选取文件</el-button>
              </div>
              <!-- 左边图片框 -->
              <div class="image-preview-1">
                  <img :src="imagePreview1" alt="请上传照片">
              </div>
            </div>
            <!-- 文件识别 -->
            <div class="right-img">
              <div class="right-file-output">
                <el-button type="success" @click="submitUpload">识别</el-button>
              </div>
              <!-- 右边图片框 -->
              <div class="image-preview-2">
                <img :src="imagePreview2" alt="待识别结果">
              </div>
            </div>
            <div class="clearfix"></div>
            <!-- 存放识别出来的结果文字 -->
            <el-alert
              v-if="recognitionResult"
              title="识别结果"
              :closable="false"
              :show-icon="true"
              type="success">
              {{ recognitionResult }}
            </el-alert>
          </div>     
        </div>
      </el-container>
    </el-container>
  </div>
</template>

<script>
export default {
  data() {
    return {
      imagePreview1: '',
      imagePreview2: '',
      recognitionResult: ''
    };
  },
  methods: {
    previewImage(event) {
      const file = event.target.files[0];
      if (file) {
        this.imagePreview1 = URL.createObjectURL(file);
      }
    },
    async submitUpload() {
      const file = this.$refs.fileInput.files[0];
      if (!file) {
        alert('请先选择文件');
        return;
      }

      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await this.$axios.post('http://localhost:8080/api/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: 30000  // 增加超时时间为30秒
        });

        if (response.status === 200) {
          const data = response.data;
          this.imagePreview2 = `data:image/jpeg;base64,${data.image}`;
          this.recognitionResult = data.prediction;
        }
      } catch (error) {
        console.error('上传出错:', error.response ? error.response.data : error);
        alert(error.response ? error.response.data.error : '上传出错');
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

.right-side-upload {
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: lightgray;
  border: 1px solid #ccc;
  width: 1228px;
}
  
.file-input {
  margin-right: 20px;
}

.image-preview-1 {
  width: 300px;
  height: 300px;
  border: 2px solid #ccc;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f0f0f0;
}

.image-preview-1 img {
  max-width: 100%;
  max-height: 100%;
}

.image-preview-2 {
  width: 300px;
  height: 300px;
  border: 2px solid #ccc;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f0f0f0;
}

.image-preview-2 img {
  max-width: 100%;
  max-height: 100%;
}

.left-img {
  float: left;
  width: 500px;
  box-sizing: border-box;
  padding: 20px;
}

.right-img {
  float: right;
  width: 500px;
  box-sizing: border-box;
  padding: 20px;
}

.clearfix::after {
  content: "";
  display: table;
  clear: both;
}

.img {
  width: 1000px;
}
</style>
