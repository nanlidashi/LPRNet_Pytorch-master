const { defineConfig } = require('@vue/cli-service')

module.exports = {
  // 基本路径
  publicPath: process.env.NODE_ENV === 'production' ? './' : '/',
  // 输出文件目录
  outputDir: 'dist',
  // 静态资源目录
  assetsDir: 'static',
  // 是否使用eslint
  lintOnSave: true,
  // 生产环境是否生成sourceMap文件
  productionSourceMap: false,
  // 跨域设置
  devServer: {
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        pathRewrite: {
          '^/api': ''
        }
      }
    }
  },
  // webpack配置
  configureWebpack: {
    resolve: {
      alias: {
        '@': require('path').resolve(__dirname, 'src/')
      }
    }
  },
  transpileDependencies: true
}
