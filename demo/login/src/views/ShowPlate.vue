<template>
  <div>
    <el-container class="main-container">
      <el-header class="top-div">
        <p>车牌识别系统</p>
      </el-header>
      <el-container class="bottom-div">
        <el-aside class="left-side">
          <el-menu default-active="3" class="el-menu-vertical-demo" background-color="#545c64" text-color="#fff" active-text-color="#ffd04b">
            <el-menu-item index="1">
              <a href="/index">首页</a>
            </el-menu-item>
            <el-menu-item index="2">
              <a href="/upload">上传测试车牌</a>
            </el-menu-item>
            <el-menu-item index="3">
              <a href="/search">展示识别车牌</a>
            </el-menu-item>
            <el-menu-item index="4">
              <a href="/ceshi">历史测试车牌</a>
            </el-menu-item>
          </el-menu>
        </el-aside>
        <el-main class="right-side-ceshi">
          <div class="sousuo">
            <input type="text" v-model="keyword" placeholder="输入搜索车牌号">
            <el-button @click="search">搜索</el-button>
          </div>
          <div class="zhanshi">
            <div v-if="searchResults.length === 0">暂无搜索结果</div>
            <div v-for="(result, index) in paginatedResults" :key="index" class="result">
              <img :src="'data:image/png;base64,' + result.base64" alt="识别车牌照片">
              <p>实际车牌号: {{ result.target }}</p>
              <p>识别车牌号: {{ result.predict }}</p>
            </div>
          </div>
          <el-pagination
            @current-change="handleCurrentChange"
            :current-page="currentPage"
            :page-size="pageSize"
            layout="prev, pager, next"
            :total="totalResults"
          ></el-pagination>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      keyword: '', // 存储搜索关键词
      searchResults: [], // 存储搜索结果
      currentPage: 1, // 当前页码
      pageSize: 15, // 每页显示数量
    };
  },
  created() {
    // 页面加载时直接获取所有数据
    this.fetchData();
  },
  computed: {
    // 计算总页数
    totalResults() {
      return this.searchResults.length;
    },
    // 分页显示的结果
    paginatedResults() {
      const start = (this.currentPage - 1) * this.pageSize;
      const end = start + this.pageSize;
      return this.searchResults.slice(start, end);
    },
  },
  methods: {
    fetchData() {
      // 发送请求获取数据
      axios.get('http://localhost:8080/api/plates')
        .then(response => {
          this.searchResults = response.data;
        })
        .catch(error => {
          console.error('Error:', error);
        });
    },
    search() {
      // 如果搜索关键词为空，则重新获取所有数据
      if (this.keyword === '') {
        this.fetchData();
        return;
      }
      // 否则根据关键词搜索
      axios.get('http://localhost:8080/api/search', { params: { keyword: this.keyword } })
        .then(response => {
          this.searchResults = response.data;
          this.currentPage = 1; // 搜索时重置页码为第一页
        })
        .catch(error => {
          console.error('Error:', error);
        });
    },
    handleCurrentChange(val) {
      this.currentPage = val;
    },
  }
}
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
  grid-template-rows: 8% 92%;
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
  background-color: rgb(209, 19, 19);
  border: 1px solid black; /* 添加边框线 */
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
  margin-bottom: 100px;
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

.right-side-ceshi {
  display: flex;
  flex-direction: column;
  padding: 20px;
  align-items: center; /* 水平居中 */
  justify-content: flex-start; /* 使内容顶部对齐 */
  background-color: white;
  border: 1px solid #ccc;
  width: 1228px;
}

.sousuo {
  width: 300px; /* 使搜索框的宽度固定 */
  margin-bottom: 20px; /* 添加底部间距 */
  display: flex; /* 使用flex布局 */
  align-items: center; /* 垂直居中 */
}

.sousuo input {
  flex: 1; /* 搜索框自动填充剩余空间 */
  height: 40px; /* 设置搜索框高度 */
}

.sousuo button {
  height: 40px; /* 设置按钮高度 */
  margin-left: 10px; /* 添加左侧间距 */
}

.zhanshi {
  width: 100%;
  padding: 20px;
  box-sizing: border-box;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  overflow-y: auto; /* 超出部分显示滚动条 */
}

.result {
  width: calc(20% - 25px); /* 每行显示五个，考虑到间距 */
  margin-bottom: 20px;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  background-color: #f9f9f9;
}



</style>
