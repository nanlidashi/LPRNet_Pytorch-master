import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

export const constantRoutes = [

  {
    path: '',
    redirect: '/login'
  },

  {
    path: '/login',
    component: () => import('@/views/login'),
    meta: {
      keepAlive: true,
      title: '登录页面'
    }
  },

  {
    path: '/register',
    component: () => import('@/views/register'),
    meta: {
      title: '注册页面'
  }
  },

  {
    path: '/index',
    component: () => import('@/views/home/AppIndex'),
    meta: {
      title: '首页'
    }
  },

  {
    path: '/upload',
    component: () => import('@/views/UpLoad'),
    meta: {
      title: '上传'
    }
  },

  {
    path: '/ceshi',
    component: () => import('@/views/CeShi'),
    meta: {
      title: '历史测试记录'
    }
  },

 
]

const createRouter = () => new Router({
  mode: 'history', // require service support
  routes: constantRoutes
});

export const router = createRouter();

router.beforeEach((to, from, next) => {
  if (to.meta.title) {
    document.title = to.meta.title;
  }
  next();
});


export function resetRouter() {
  const newRouter = createRouter()
  router.matcher = newRouter.matcher // reset router
}

export default router
