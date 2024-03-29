import Vue from 'vue'
import Router from 'vue-router'
import AppIndex from '@/components/home/AppIndex.vue'
import DeLu from "@/components/DeLu.vue";
import UpLoad from '@/components/UpLoad.vue';
import CeShi from '@/components/CeShi.vue';
import ZhuCe from '@/components/ZhuCe.vue';

Vue.use(Router)

const router = new Router({
    mode: 'history',
    routes: [
        {
            path: '/',
            redirect: '/login',  // 默认重定向到/login
            meta: {
                title: '登录页面'
            }
        },
        {
            path: '/login',
            name: 'DeLu',
            component: DeLu,
            meta: {
                title: '登录页面'
            }
        },
        {
            path: '/register',
            name: 'ZhuCe',
            component: ZhuCe,
            meta: {
                title: '注册页面'
            }
        },
        {
            path: '/index',
            name: 'AppIndex',
            component: AppIndex,
            meta: {
                title: '首页'
            }
        },
        {
            path: '/upload',
            name: 'UpLoad',
            component: UpLoad,
            meta: {
                title: '上传'
            }
        },
        {
            path: '/ceshi',
            name: 'CeShi',
            component: CeShi,
            meta: {
                title: '历史测试记录'
            }
        }
    ]
});

router.beforeEach((to, from, next) => {
    // 移除这个条件，以允许正常导航
    // if (to.path === from.path) {
    //     next(false);
    // } else {
    //     next();
    // }
    // 设置页面标题
    document.title = to.meta.title || '默认标题';
    next(); // 直接进行正常导航
});

export default router;
