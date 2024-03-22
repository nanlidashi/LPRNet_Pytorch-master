import Vue from 'vue'
import Router from 'vue-router'
import AppIndex from '@/components/home/AppIndex.vue'
import DeLu from "@/components/DeLu.vue";

Vue.use(Router)

const router = new Router({
    mode: 'history',
    routes: [
        {
            path: '/login',
            name: 'DeLu',
            component: DeLu
        },
        {
            path: '/index',
            name: 'AppIndex',
            component: AppIndex
        }
    ]
});

router.beforeEach((to, from, next) => {
    // 检查当前路由是否已经是即将导航到的路由
    if (to.path === from.path) {
        // 如果是，不进行导航
        next(false);
    } else {
        // 如果不是，进行正常导航
        next();
    }
});

export default router;
