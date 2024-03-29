<template>
  <div>
    <el-container class="main-container">
      <el-header class="top-div">
        <p>车牌识别系统</p>
      </el-header>
      <el-container class="bottom-div">
        <el-aside class="left-side">
          <el-menu default-active="1" class="el-menu-vertical-demo" background-color="#545c64" text-color="#fff" active-text-color="#ffd04b">
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
            <el-col>
              <div class="intro-container">
                <h2 class="intro-title">LPRNet车牌识别系统介绍：</h2>
                <p class="intro-content" style="color: #f00;">
                  LPRNet（License Plate Recognition Network）是一种先进的车牌识别技术，它采用深度学习方法进行车牌号码的自动检测和识别。该系统结合了图像处理和深度学习技术，能够在复杂的环境中准确快速地识别出车辆的车牌号码。
                </p>
                <h3 class="workflow-title">系统工作流程：</h3>
                <ul class="workflow-list">
                  <li style="color: #f00;">1. 用户通过前端界面上传车辆图片。</li>
                  <li style="color: #f00;">2. 前端将图片发送到后端。</li>
                  <li style="color: #f00;">3. 后端使用LPRNet算法进行车牌识别。</li>
                  <li style="color: #f00;">4. 后端返回识别结果给前端。</li>
                  <li style="color: #f00;">5. 前端显示识别结果。</li>
                </ul>
                <h3 class="feature-title">技术特点：</h3>
                <ul class="feature-list">
                  <li style="color: #f00;">高准确性：采用深度学习模型，具有高度的识别准确性。</li>
                  <li style="color: #f00;">高效性能：快速处理车辆图片，实时返回识别结果。</li>
                  <li style="color: #f00;">稳定可靠：系统稳定性高，能够在各种环境下进行稳定识别。</li>
                </ul>
                <p class="conclusion" style="color: #2567ac;">
                  采用LPRNet的车牌识别系统为交通管理、停车场管理、安全监控等领域提供了高效、准确的解决方案。
                </p>
                <h3 class="usage-title">使用方法：</h3>
                <ul class="usage-list">
                  <li style="color: #f00;">1. 选择“上传测试车牌”进入上传页面。</li>
                  <li style="color: #f00;">2. 点击“选取文件”按钮选择要识别的车牌图片。</li>
                  <li style="color: #f00;">3. 点击“识别”按钮进行车牌识别。</li>
                  <li style="color: #f00;">4. 识别完成后，系统将展示识别结果。</li>
                </ul>
                <div class="separator"></div>
                <p class="note" style="color: #2567ac;">注意：请确保上传的图片为清晰可见的车牌图片，以获得最佳识别效果。</p>
              </div>
            </el-col>
          </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script>

export default {
  name: "AppIndex",
  data() {
    return {
      message: ''
    };
  },
  created() {
    this.fetchHello();
  },
  methods: {
    async fetchHello() {
      try {
        const response = await this.$axios.get('/index');
        this.message = response.data;
      } catch (error) {
        console.error('Error fetching hello:', error);
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
      height: 100vh; /* 100% 视口高度 */
      display: grid;
      grid-template-rows: 20% 80%; /* 上下两个 div 的高度比例 */
  }

  .top-div {
      background-color: lightblue;
      border: 1px solid #ccc; /* 添加边框 */
      display: flex;
      justify-content: center; /* 水平居中 */
      align-items: center; /* 垂直居中 */
  }

  .top-div p {
      margin: 0; /* 移除 p 标签的默认边距 */
  }

  .bottom-div {
      display: grid;
      grid-template-columns: 1fr 1fr; /* 两列，每列等宽 */
  }

  .left-side {
      background-color: lightgray;
      border: 1px solid #ccc; /* 添加边框 */
      width: 300px; /* 左右两边各占 20% 的宽度 */
      background-color: rgb(209, 19, 19);
      border: 1px solid black; /* 添加边框线 */
      padding: 10px; /* 内边距 */
  }

  .left-side ul {
      list-style-type: none; /* 移除默认的列表样式 */
      padding: 0; /* 移除默认的内边距 */
      justify-content: center;  /* 垂直居中 */
      flex-direction: column; /* 垂直排列 */
  }

  .left-side li {
      font-size: 18px; /* 设置字体大小 */
      margin-bottom: 100px; /* 设置列表项之间的间距为 15px */
      cursor: pointer; /* 鼠标悬停时显示手型 */
      transition: 0.3s; /* 平滑过渡效果 */
      text-align: center; /* 文字左右居中 */
  }

  .left-side li a {
      text-decoration: none; /* 取消下划线 */
      color: inherit; /* 继承父元素颜色 */
  }

  .left-side li:hover {
      background-color: rgba(255, 255, 255, 0.2); /* 鼠标悬停时背景色变浅 */
      border-radius: 4px; /* 设置圆角 */
  }

  .right-side {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    background-image: url('../../assets/index-1.jpg');  /*设置背景图片 */
    background-size: cover;  /*调整背景图片大小以覆盖整个容器 */
    background-repeat: no-repeat;  /*禁止背景图片重复 */
    background-position: center;  /*设置背景图片位置 */
    /* background-color: rgb(218, 6, 6); */
    border: 1px solid #08d36e; /* 添加边框 */
    width:1000px;
    height: 700px;
}
.intro-container {
    max-width: 800px;
    margin-top: 30px;
  }

  .intro-title, .workflow-title, .feature-title, .usage-title {
    font-size: 28px;
    font-weight: bold;
    color: #333;
    margin-bottom: 20px;
  }

  .intro-content, .workflow-list, .feature-list, .conclusion, .usage-list, .note {
    font-size: 18px;
    color: #666;
    line-height: 1.6;
    margin-bottom: 20px;
  }

  .workflow-list, .feature-list, .usage-list {
    list-style-type: disc;
    padding-left: 20px;
  }

  .separator {
    height: 2px;
    background-color: #ddd;
    margin: 30px 0;
  }

  .note {
    font-style: italic;
    color: #777;
  }

</style>