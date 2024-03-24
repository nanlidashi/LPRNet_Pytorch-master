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
              <div>
                <p class="message">{{ message }}</p>
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
    background-image: url('https://picsum.photos/1000/800.jpg'); /* 设置背景图片 */
    background-size: cover; /* 调整背景图片大小以覆盖整个容器 */
    background-repeat: no-repeat; /* 禁止背景图片重复 */
    background-position: center; /* 设置背景图片位置 */
    background-color: rgb(218, 6, 6);
    border: 1px solid #08d36e; /* 添加边框 */
    width:1000px;
}
.message {
    font-size: 24px; /* 设置字体大小为24px */
}


</style>