const nodemailer = require('nodemailer');
const express = require('express');
const router = express.Router();

router.post('/sendEmail', async (req, res) => {
  const { email } = req.body;

  const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: '1139414320@qq.com', // 你的 Gmail 邮箱
      pass: '1274645510zjs' // 你的 Gmail 密码
    }
  });

  const code = Math.floor(1000 + Math.random() * 9000); // 生成一个随机的 4 位数验证码

  const mailOptions = {
    from: '1139414320@qq.com',
    to: email,
    subject: '验证码',
    text: `您的验证码是：${code}`
  };

  try {
    await new Promise((resolve, reject) => {
      transporter.sendMail(mailOptions, (error) => {
        if (error) {
          console.log(error);
          reject(error);
        } else {
          resolve();
        }
      });
    });
    res.status(200).send({ message: '邮件发送成功', code: code });
  } catch (error) {
    console.log(error);
    res.status(500).send({ message: '邮件发送失败' });
  }
});

module.exports = router;
