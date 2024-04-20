<template>
  <div class="canvas">
    <canvas id="canvas" class="validCode"></canvas>
  </div>
</template>

<script>
export default {
  name: 'ValidCode',
  props: {
    identifyCode: { 
      type: String,
      default: '1234'
    },
    fontSizeMin: { 
      type: Number,
      default: 130
    },
    fontSizeMax: { 
      type: Number,
      default: 140
    }
  },
  methods: {
    // 生成一个随机数
    randomNum (min, max) {
      return Math.floor(Math.random() * (max - min) + min)
    },

    // 生成一个随机的颜色
    randomColor (min, max) {
      const r = this.randomNum(min, max)
      const g = this.randomNum(min, max)
      const b = this.randomNum(min, max)
      return 'rgb(' + r + ',' + g + ',' + b + ')'
    },

    drawPic () {
      const canvas = document.getElementById('canvas')
      const ctx = canvas.getContext('2d')
      ctx.textBaseline = 'bottom'
      // console.log(canvas.width)
      // 绘制背景
      ctx.fillStyle = '#e6ecfd'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      // 绘制文字
      for (let i = 0; i < this.identifyCode.length; i++) {
        this.drawText(ctx, this.identifyCode[i], i)
      }
      this.drawLine(ctx)
      this.drawDot(ctx)
    },
    drawText (ctx, txt, i) {
      const canvas = document.getElementById('canvas')
      ctx.fillStyle = this.randomColor(50, 160) // 随机生成字体颜色
      ctx.font = this.randomNum(this.fontSizeMin, this.fontSizeMax) + 'px SimHei' // 随机生成字体大小
      const x = (i + 0.5 ) * (canvas.width / (this.identifyCode.length + 1))
      const y = this.randomNum(this.fontSizeMax, canvas.height - 5)
      var deg = this.randomNum(-30, 30)
      // 修改坐标原点和旋转角度
      ctx.translate(x, y)
      ctx.rotate(deg * Math.PI / 180)
      ctx.fillText(txt, 0, 0)
      // 恢复坐标原点和旋转角度
      ctx.rotate(-deg * Math.PI / 180)
      ctx.translate(-x, -y)
    },

    drawLine (ctx) {
      // 绘制干扰线
      const canvas = document.getElementById('canvas')
      for (let j = 0; j < 4; j++) {
        ctx.strokeStyle = this.randomColor(100, 200)
        ctx.beginPath()
        ctx.moveTo(this.randomNum(0, canvas.width), this.randomNum(0, canvas.height))
        ctx.lineTo(this.randomNum(0, canvas.width), this.randomNum(0, canvas.height))
        ctx.lineWidth = 3
        ctx.stroke()
      }
    },

    drawDot (ctx) {
      // 绘制干扰点
      const canvas = document.getElementById('canvas')
      for (let k = 0; k < 30; k++) {
        ctx.fillStyle = this.randomColor(0, 255)
        ctx.beginPath()
        ctx.arc(this.randomNum(0, canvas.width), this.randomNum(0, canvas.height), 3, 0, 2 * Math.PI)
        ctx.fill()
      }
    }
  },
  watch: {
    identifyCode () {
      this.drawPic()
    }
  },
  mounted () {
    this.drawPic()
  }
}
</script>

<style scoped>
  .canvas{
    height: 38px;
  }
  .validCode{
    width: 113px;
    height: 38px;
  }
</style>